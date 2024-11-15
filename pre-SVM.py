import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from scipy.stats import kurtosis, skew
from sklearn.impute import SimpleImputer
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
    
    # 添加空值处理
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
    
    # 确保没有NaN值
    if features.isnull().any().any():
        features = features.fillna(features.mean())
    
    print("特征形状:", features.shape)
    return features.T.to_numpy()

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

# 方法1：网格搜索
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']  # 添加linear核函数
}

grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,                # 5折交叉验证
    scoring='accuracy',  # 评分标准
    n_jobs=-1,          # 使用所有CPU核心
    verbose=2           # 输出训练过程
)

# 训练网格搜索模型
grid_search.fit(X_train, y_train)

# 输出最佳参数和得分
print(f'最佳参数: {grid_search.best_params_}')
print(f'最佳得分: {grid_search.best_score_:.4f}')

# 使用最佳参数的模型
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 计算测试集准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'测试集准确率: {accuracy:.4f}')

# 方法2：随机搜索（更快，但可能不如网格搜索精确）
param_distributions = {
    'C': np.logspace(-3, 3, 7),           # 生成7个在10^-3到10^3之间的对数均匀分布的数
    'gamma': np.logspace(-3, 2, 6),       # 生成6个在10^-3到10^2之间的对数均匀分布的数
    'kernel': ['rbf']
}

random_search = RandomizedSearchCV(
    SVC(),
    param_distributions,
    n_iter=20,          # 随机尝试20种参数组合
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# 训练随机搜索模型
random_search.fit(X_train, y_train)

# 输出最佳参数和得分
print(f'最佳参数: {random_search.best_params_}')
print(f'最佳得分: {random_search.best_score_:.4f}')

# 使用最佳参数的模型
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# 计算测试集准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'测试集准确率: {accuracy:.4f}')

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
save_directory = 'H:/project/SVM/figures'
plt.savefig(os.path.join(save_directory, 'pre_混淆矩阵.png'))
plt.show()

# 可视化特征分布（示例：前两个特征）
plt.figure(figsize=(10, 5))
plt.scatter(features[:, 0], features[:, 1], c=encoded_labels, cmap='viridis', alpha=0.5)
plt.title('Feature Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Label')
plt.savefig(os.path.join(save_directory, 'pre_feature.png'))
plt.show()