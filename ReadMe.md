CNN 方法
目前最好：2-Mers; 130 epoch（2 epoch）; test accuracy 72.37%
--k 2 --hidden_size 330 --epochs 2 --network RNA_Classifier_2;; test accuracy 73.95% 效果一直尚可稳定！


分层采样：--k 3 --batch_size 32 --epoch 4 test accuracy 70.53%
        --k 4 --batch_size 32 --epoch 5 test accuracy 72.63%

Adaboost 方法
讨论depth, n_estimators, k 之间的关系
depth and k 增大，都会使得更快收敛；
n_estimators过多会导致过拟合，效果不好

depth = 1
n_estimaotors=500; k=2; learning_rate = 0.1; validation accuracy 67.64%; test accuracy 71.05%
n_estimaotors=500; k=2; learning_rate = 0.05; validation accuracy 67.20%; test accuracy 72.37%
n_estimaotors=500; k=3; learning_rate = 0.05; validation accuracy 65.45%; test accuracy 66.32%
n_estimaotors=250; k=3; learning_rate = 0.05; validation accuracy 65.59%; test accuracy 69.74%
n_estimaotors=125; k=3; learning_rate = 0.05; validation accuracy 66.76%; test accuracy 70.26%
n_estimaotors=100; k=3; learning_rate = 0.05; validation accuracy 67.64%; test accuracy 69.47%(table中0 presision 达到了0.75 高！)
n_estimaotors=80; k=3; learning_rate = 0.05; validation accuracy 66.62%; test accuracy 68.68%

depth = 2
n_estimaotors=200; k=2; learning_rate = 0.05; validation accuracy 66.76%%; test accuracy 72.37%


depth = 3
n_estimaotors=100; k=2; learning_rate = 0.05; validation accuracy 66.18%%; test accuracy 72.37%


XGBoost方法：
n_estimaotors=500; k=2; learning_rate = 0.05; validation accuracy 67.79%; test accuracy 72.63%


SVM方法：(validation accuracy & test accuracy)
k = 5, Kernel: rbf 67.35/68.16
k = 4, Kernel: rbf 68.67/70.00
k = 3, Kernel: rbf 69.40/70.00; linear 68.52/69.21; poly 65.74/67.89; sigmoid 59.15/57.37
k = 2, Kernel: rbf 68.37/70.00
k = 1, Kernel: rbf 59.88/62.11