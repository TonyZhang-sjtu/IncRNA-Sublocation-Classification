import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 需要导入itertools中的product来生成所有可能的k-mer组合
from itertools import product

# K-mer编码
# def kmer_encoding(sequence, k=3):
#     """K-mer频率编码"""
#     # 可能的k-mer（4个碱基的k次幂）
#     kmers = [''.join([a, b, c]) for a in 'ACGU' for b in 'ACGU' for c in 'ACGU']
#     kmer_dict = {kmer: 0 for kmer in kmers}
    
#     # 生成K-mer频率向量
#     for i in range(len(sequence) - k + 1):
#         kmer = sequence[i:i + k]
#         if kmer in kmer_dict:
#             kmer_dict[kmer] += 1
    
#     # 转换为频率
#     kmer_freq = np.array(list(kmer_dict.values()))
#     return kmer_freq / np.sum(kmer_freq)  # 归一化


def kmer_encoding(sequence, k=3):
    """K-mer频率编码，支持动态k值"""
    # 获取所有可能的K-mer（基于ACGU四个碱基）
    kmers = [''.join(x) for x in product('ACGU', repeat=k)]  # 动态生成k-mer组合
    kmer_dict = {kmer: 0 for kmer in kmers}
    
    # 生成K-mer频率向量
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmer_dict:
            kmer_dict[kmer] += 1
    
    # 转换为频率
    kmer_freq = np.array(list(kmer_dict.values()))
    return kmer_freq / np.sum(kmer_freq)  # 归一化



# 定义神经网络
class RNAClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(RNAClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

# 主要分类过程
def classification(training_file, test_file, k=3, hidden_size=128, batch_size=32, epochs=20):
    # 读取数据
    train_data = pd.read_csv(training_file)
    test_data = pd.read_csv(test_file)

    # 提取RNA序列和标签
    X_train = [kmer_encoding(seq, k) for seq in train_data.iloc[:, 2]]  # 3列为RNA序列
    y_train = train_data.iloc[:, 3].values  # 4列为标签
    
    X_test = [kmer_encoding(seq, k) for seq in test_data.iloc[:, 2]]
    y_test = test_data.iloc[:, 3].values
    
    # 转换为numpy数组
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 转换为torch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # 初始化神经网络
    input_size = X_train.shape[1]  # K-mer的数量（例如3-mer是64）
    model = RNAClassifier(input_size, hidden_size)
    
    # 损失函数和优化器
    criterion = nn.BCELoss()  # 二分类损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            # 获取一个batch的数据
            inputs = X_train_tensor[i:i+batch_size]
            labels = y_train_tensor[i:i+batch_size]
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 打印每个epoch的损失
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # 测试模型
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred = (y_pred > 0.5).float()  # 二分类输出0或1
        
        accuracy = (y_pred == y_test_tensor).float().mean()
        print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
    
    return model

# 主函数调用
def main():
    training_file = 'data/lncRNA_sublocation_Training&Validation.csv'
    test_file = 'data/lncRNA_sublocation_Test.csv'
    
    # 调用分类函数
    classification(training_file, test_file, k=2, hidden_size=128, epochs=500)

if __name__ == "__main__":
    main()
