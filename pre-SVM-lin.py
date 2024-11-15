import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")
# 加载数据
data = pd.read_csv('H:/project/SVM/data/combined_data/pre_combined_data.csv', header=None)
print("first row:", data.iloc[0, :].values)

# 假设第一行是标签
labels = data.iloc[0, :].values  # 提取第一行的所有列作为标签
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 从第二行开始提取特征
def extract_features(data):
    # 从第二行开始提取数据
    numeric_data = data.iloc[1:, :].apply(pd.to_numeric, errors='coerce')
    # 使用均值和标准差作为特征
    features = numeric_data.apply(lambda col: [col.mean(), col.std()], axis=0)
    print("shape:", features.shape)
    return features.T.to_numpy()  # 转置后转换为 NumPy 数组

# 提取特征
features = extract_features(data)

# 确保标签数量与特征数量一致
if len(encoded_labels) != features.shape[0]:
    raise ValueError("标签数量与特征数量不一致，请检查数据结构。")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练SVM模型
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'识别准确率: {accuracy * 100:.2f}%')

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print('混淆矩阵:')
print(conf_matrix)

# 可视化混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 可视化特征分布（示例：前两个特征）
plt.figure(figsize=(10, 5))
plt.scatter(features[:, 0], features[:, 1], c=encoded_labels, cmap='viridis', alpha=0.5)
plt.title('Feature Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Label')
plt.show()