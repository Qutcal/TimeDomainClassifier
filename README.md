# 时域特征分类器

基于摩擦起电与静电感应效应的触觉传感器在接触不同材料的时候会输出不同的信号，本项目基于触觉传感器在不同材料上采集的信号实现了基于时域特征的分类模型，使用了SVM、LSTM以及TCN进行对比实验。

## 特征提取

所有模型使用相同的时域特征提取方法，包括：
- 均值
- 标准差
- 最大值
- 最小值
- 峰峰值
- RMS值
- 偏度
- 峭度
- 四分位距
- 中位数

## 数据预处理 (preprocess.ipynb)

### 数据清洗流程
1. 数据加载与初始检查
- 读取原始CSV文件
- 检查数据维度和基本结构
- 可视化原始数据分布

2. 标签处理
- 为每个数据文件添加对应的材料类型标签
- 标签包括：A4纸、棉布、泡沫、玻璃、皮革、PTFE、羊毛、木材、铜片

3. 数据清理
- 去除缺失值
- 移除无效数据段
- 处理数据突变部分
- 使用阈值过滤异常值

4. 数据合并
- 将不同材料的数据文件合并
- 保持标签与数据的对应关系
- 确保数据格式统一

### 输出文件（请自行创建）
- combined_data.csv: 原始合并数据
- pre_combined_data.csv: 预处理后的合并数据

## 模型实现

### 1. TCN模型 (time_domain_tcn.py)
- 使用残差块构建的时间卷积网络
- 包含空洞卷积和残差连接
- 批归一化和dropout用于防止过拟合

### 2. LSTM模型 (time_domain_lstm.py)
- 双层LSTM网络结构
- dropout层防止过拟合
- 全连接层进行最终分类

### 3. SVM模型 (pre-SVM.py)
- 使用网格搜索和随机搜索优化超参数
- 支持RBF和线性核函数
- 交叉验证确保模型稳定性

### 4. 线性SVM模型 (pre-SVM-lin.py)
- 简化版SVM实现
- 仅使用线性核函数
- 适用于快速测试和基准对比

## 使用方法

1. 数据准备：
```python
data = pd.read_csv('path/to/your/data.csv', header=None)
```

2. 模型训练：
```python
# 运行对应的Python文件即可
python time_domain_tcn.py
python time_domain_lstm.py
python pre-SVM.py
python pre-SVM-lin.py
```

## 输出结果

所有模型都会输出：
- 训练过程中的损失和准确率
- 最终测试准确率
- 混淆矩阵可视化
- 特征分布图（SVM模型）

## 环境要求

### Python版本
- Python 3.11


### 安装依赖
1. 使用requirements.txt安装
```bash
pip install -r requirements.txt
```

2. 或手动安装各个包
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch scipy jupyter
```

## 注意事项

- 所有警告信息已被禁用
- 使用了固定的随机种子(42)确保结果可复现
