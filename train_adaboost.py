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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score


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


# 分类函数
def classification(training_file, test_file, k=3, n_estimators=50, learning_rate=1.0, max_depth=1):
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
    
    # 定义弱分类器（决策树）
    base_estimator = DecisionTreeClassifier(max_depth=max_depth)
    
    # 初始化Adaboost分类器
    model = AdaBoostClassifier(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )
    
    # 训练模型
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

    # 可视化特征重要性
    feature_importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importances)), feature_importances)
    plt.xlabel('K-mer Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance from Adaboost')
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
    parser.add_argument('--n_estimators', type=int, default=50, help='Number of estimators for Adaboost')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for Adaboost')
    parser.add_argument('--max_depth', type=int, default=1, help='Max depth of each decision tree')
    parser.add_argument('--cv', type=int, default=0, help='Whether to select hyper parameter n_estimators by corss validation')


    # 解析命令行参数
    args = parser.parse_args()
    # 使用cv选择n_estimators
    if args.cv:
        # 定义 n_estimators 的搜索范围
        n_estimators_range = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

        # 调用超参数选择函数
        best_n_estimators, results = select_n_estimators(args.training_file, args.test_file, args.k, n_estimators_range, cv=5, scoring='accuracy')

        # 打印最佳结果
        print(f"Optimal n_estimators: {best_n_estimators}")

        # 调用分类函数
        classification(
            args.training_file,
            args.test_file,
            k=args.k,
            n_estimators=best_n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth
        )
    
    # 给定n_estimators
    else:
        # 调用分类函数
        classification(
            args.training_file,
            args.test_file,
            k=args.k,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth
        )

if __name__ == "__main__":
    main()

