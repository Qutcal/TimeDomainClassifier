import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import kurtosis, skew
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import warnings 
# 忽略所有警告
warnings.filterwarnings("ignore")
# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        
        # 修改填充计算方式
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # 1x1 卷积用于通道数调整
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        # 添加批归一化
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        # 确保残差连接的维度匹配
        if out.size(2) != residual.size(2):
            # 调整残差的长度以匹配输出
            diff = out.size(2) - residual.size(2)
            if diff > 0:
                residual = torch.nn.functional.pad(residual, (0, diff))
            else:
                out = torch.nn.functional.pad(out, (0, -diff))
        
        return self.relu(out + residual)

# 定义TCN模型
class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout, num_classes):
        super(TCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size, dilation))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
    
    def forward(self, x):
        out = self.network(x)
        out = torch.mean(out, dim=2)  # 全局平均池化
        return self.fc(out)

# 自定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 特征提取函数
def extract_features(data):
    numeric_data = data.iloc[1:, :].apply(pd.to_numeric, errors='coerce')
    
    imputer = SimpleImputer(strategy='mean')
    numeric_data = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)
    
    def calculate_features(col):
        return [
            col.mean(),          # 均值
            col.std(),           # 标准差
            col.max(),           # 最大值
            col.min(),           # 最小值
            col.max() - col.min(), # 峰峰值
            np.sqrt(np.mean(col**2)), # RMS值
            skew(col),           # 偏度
            kurtosis(col),       # 峭度
            np.percentile(col, 75) - np.percentile(col, 25), # 四分位距
            np.median(col)       # 中位数
        ]
    
    features = numeric_data.apply(calculate_features, axis=0)
    
    if features.isnull().any().any():
        features = features.fillna(features.mean())
    
    print("特征形状:", features.shape)
    return features.T.to_numpy()

# 训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# 评估函数
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    return test_loss, test_acc, all_predictions, all_labels

def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    data = pd.read_csv('H:/project/SVM/data/combined_data/pre_combined_data.csv', header=None)
    print("数据形状:", data.shape)
    
    # 提取标签和特征
    labels = data.iloc[0, :].values
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    features = extract_features(data)
    
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 重塑数据为TCN所需的格式 (batch_size, channels, sequence_length)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # 创建数据加载器
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 模型参数
    input_size = X_train.shape[1]  # 特征维度
    num_channels = [32, 64, 128]   # 减少通道数
    kernel_size = 3                # 使用较小的核大小
    dropout = 0.2
    num_classes = len(np.unique(encoded_labels))
    
    # 初始化模型
    model = TCN(input_size, num_channels, kernel_size, dropout, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练参数
    num_epochs = 100
    best_acc = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, predictions, true_labels = evaluate_model(
            model, test_loader, criterion, device
        )
        
        if test_acc > best_acc:
            best_acc = test_acc
            # 保存最佳模型
            torch.save(model.state_dict(), 'H:/project/SVM/models/best_tcn_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load('H:/project/SVM/models/best_tcn_model.pth'))
    _, final_acc, final_predictions, final_true_labels = evaluate_model(
        model, test_loader, criterion, device
    )
    
    # 计算并显示混淆矩阵
    conf_matrix = confusion_matrix(final_true_labels, final_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('H:/project/SVM/figures/tcn_confusion_matrix.png')
    plt.show()
    
    print(f'最终测试准确率: {final_acc:.2f}%')

if __name__ == '__main__':
    main() 