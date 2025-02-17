import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# K-mer编码函数（支持动态k值）
from itertools import product
import argparse
import random

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# 假设RNAClassifier是你定义的神经网络模型
class RNA_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNA_Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)


class RNA_Classifier_2(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.5):
        super(RNA_Classifier_2, self).__init__()
        
        # 定义网络结构
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一层全连接
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)  # 第一层 BatchNorm
        self.dropout1 = nn.Dropout(p=dropout_prob)    # 第一层 Dropout
        
        self.fc2 = nn.Linear(hidden_size, 1)          # 第二层全连接
        self.sigmoid = nn.Sigmoid()                  # 输出层激活函数

    def forward(self, x):
        # 前向传播
        x = self.fc1(x)                              # 第一层全连接
        x = self.batchnorm1(x)                       # 第一层 BatchNorm
        x = torch.relu(x)                            # 激活函数 ReLU
        x = self.dropout1(x)                         # 第一层 Dropout
        
        x = self.fc2(x)                              # 第二层全连接
        return self.sigmoid(x)                       # 输出层激活


# 定义神经网络模型
class RNA_Classifier_3(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.5):
        super(RNA_Classifier_3, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


def kmer_encoding(sequence, k=3):
    """K-mer频率编码，支持动态k值"""
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

# 主要分类过程
def classification(training_file, test_file, k=3, hidden_size=128, batch_size=32, epochs=20, network='RNA_Classifier'):
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
    
    # 划分训练集和验证集
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # 划分训练集和验证集，并保持标签分布一致（分层采样）
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # 转换为torch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # 初始化神经网络
    input_size = X_train.shape[1]  # K-mer的数量（例如3-mer是64）
    if network == 'RNA_Classifier':
        model = RNA_Classifier(input_size, hidden_size)
    elif network == 'RNA_Classifier_2':
        model = RNA_Classifier_2(input_size, hidden_size)
    else:
        model = RNA_Classifier_3(input_size, hidden_size)
    
    # 损失函数和优化器
    criterion = nn.BCELoss()  # 二分类损失 交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # 用于记录损失和准确率
    train_loss_history = []
    val_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []
    
    # 训练模型
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
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
            
            running_loss += loss.item()
            
            # 计算训练集准确率
            predicted = (outputs > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # 计算训练集的平均损失和准确率
        train_loss = running_loss / len(X_train)
        train_accuracy = 100 * correct_train / total_train
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        
        # 验证集评估
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_predicted = (val_outputs > 0.5).float()
            correct_val = (val_predicted == y_val_tensor).sum().item()
            total_val = y_val_tensor.size(0)
            val_accuracy = 100 * correct_val / total_val
            val_loss_history.append(val_loss)
            val_accuracy_history.append(val_accuracy)
        
        # 打印每个epoch的训练和验证集的损失和准确率
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    # 测试模型
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred = (y_pred > 0.5).float()  # 二分类输出0或1
        
        accuracy = (y_pred == y_test_tensor).float().mean()
        print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
    
    # 绘制训练过程中的损失和准确率图
    epochs_range = range(1, epochs + 1)
    
    plt.figure(figsize=(12, 6))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss_history, label='Train Loss')
    plt.plot(epochs_range, val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss per Epoch')
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracy_history, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracy_history, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy per Epoch')
    
    plt.tight_layout()
    plt.show()
    
    return model


# 主函数调用
def main():
    set_seed(42)

    # 使用 argparse 处理命令行参数
    parser = argparse.ArgumentParser(description='RNA Classification using K-mer encoding')
    
    # 添加命令行参数
    parser.add_argument('--training_file', type=str, default='data/lncRNA_sublocation_Training&Validation.csv',
                        help='Path to the training dataset')
    parser.add_argument('--test_file', type=str, default='data/lncRNA_sublocation_Test.csv',
                        help='Path to the test dataset')
    parser.add_argument('--k', type=int, default=2, help='Length of k-mer encoding')
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of neurons in the hidden layer')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples in a batch')
    parser.add_argument('--network', type=str, default='RNA_Classifier', help='Type of CNN Network')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用分类函数
    classification(args.training_file, args.test_file, k=args.k, hidden_size=args.hidden_size, batch_size=args.batch_size, epochs=args.epochs, network=args.network)


if __name__ == "__main__":
    main()

