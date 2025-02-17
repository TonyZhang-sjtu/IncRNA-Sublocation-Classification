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
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


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


def select_n_estimators(training_file, test_file, k, n_estimators_range, cv=5, scoring='accuracy'):
    """
    使用交叉验证选择最优的 n_estimators 超参数。

    参数：
    - X_train: numpy.ndarray，训练集特征。
    - y_train: numpy.ndarray，训练集标签。
    - n_estimators_range: list，待搜索的 n_estimators 值范围。
    - cv: int，交叉验证折数，默认为 5。
    - scoring: str，交叉验证的评价指标，默认为 'accuracy'。

    返回：
    - best_n_estimators: int，最优的 n_estimators 值。
    - results: dict，记录每个 n_estimators 的交叉验证得分。
    """
    # 读取数据
    train_data = pd.read_csv(training_file)
    test_data = pd.read_csv(test_file)

    # 提取RNA序列和标签
    X_train = [kmer_encoding(seq, k) for seq in train_data.iloc[:, 2]]  # 第3列为RNA序列
    y_train = train_data.iloc[:, 3].values  # 第4列为标签
    
    X_test = [kmer_encoding(seq, k) for seq in test_data.iloc[:, 2]]
    y_test = test_data.iloc[:, 3].values
    
    # 转换为numpy数组
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 划分训练集和验证集，并保持标签分布一致
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    

    results = {}
    for n_estimators in n_estimators_range:
        print(f"Evaluating n_estimators={n_estimators}...")
        model = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=n_estimators,
            random_state=42
        )
        # 使用 cross_val_score 进行交叉验证
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        results[n_estimators] = np.mean(scores)
        print(f"Mean CV Accuracy for n_estimators={n_estimators}: {results[n_estimators]:.4f}")
    
    # 找到最佳 n_estimators
    best_n_estimators = max(results, key=results.get)
    print(f"\nBest n_estimators: {best_n_estimators} with Mean CV Accuracy: {results[best_n_estimators]:.4f}")
    return best_n_estimators, results


# 分类函数 - 使用 SVM
def classification_with_svm(training_file, test_file, k=3, kernel='rbf', C=1.0, gamma='scale'):
    """
    使用支持向量机 (SVM) 对 RNA 序列进行分类。
    
    参数：
    - training_file: str，训练数据的路径。
    - test_file: str，测试数据的路径。
    - k: int，K-mer 的长度。
    - kernel: str，SVM 核函数类型。
    - C: float，正则化参数。
    - gamma: str 或 float，核函数系数。
    """
    # 读取数据
    train_data = pd.read_csv(training_file)
    test_data = pd.read_csv(test_file)

    # 提取RNA序列和标签
    X_train = [kmer_encoding(seq, k) for seq in train_data.iloc[:, 2]]  # 第3列为RNA序列
    y_train = train_data.iloc[:, 3].values  # 第4列为标签
    
    X_test = [kmer_encoding(seq, k) for seq in test_data.iloc[:, 2]]
    y_test = test_data.iloc[:, 3].values
    
    # 转换为numpy数组
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 划分训练集和验证集，并保持标签分布一致
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # 初始化 SVM 模型
    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
    
    # 训练 SVM 模型
    model.fit(X_train, y_train)
    
    # 验证集评估
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    
    # 测试集评估
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # 输出详细分类报告
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    return model


# 主函数调用
def main():
    set_seed(42)

    # 使用 argparse 处理命令行参数
    parser = argparse.ArgumentParser(description='RNA Classification using K-mer encoding with SVM')
    
    # 添加命令行参数
    parser.add_argument('--training_file', type=str, default='data/lncRNA_sublocation_Training&Validation.csv',
                        help='Path to the training dataset')
    parser.add_argument('--test_file', type=str, default='data/lncRNA_sublocation_Test.csv',
                        help='Path to the test dataset')
    parser.add_argument('--k', type=int, default=2, help='Length of k-mer encoding')
    parser.add_argument('--kernel', type=str, default='rbf', help='SVM kernel type (e.g., linear, poly, rbf, sigmoid)')
    parser.add_argument('--C', type=float, default=1.0, help='Regularization parameter for SVM')
    parser.add_argument('--gamma', type=str, default='scale', help='Kernel coefficient for SVM (scale, auto, or float)')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用 SVM 分类函数
    classification_with_svm(
        args.training_file,
        args.test_file,
        k=args.k,
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma
    )


if __name__ == "__main__":
    main()
