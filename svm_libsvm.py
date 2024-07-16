from libsvm.svmutil import *
import time

t0 = time.time()
# 读取数据
Y, X = svm_read_problem('german.numer_scale.txt')
#70%的数据用于训练，30%的数据用于测试
train_size = int(0.7 * len(Y))
x_train = X[:train_size]
y_train = Y[:train_size]
x_test = X[train_size:]
y_test = Y[train_size:]


#设置参数，使用RBF核

param = '-t 2 -c 10 -b 0 -g 0.041666666666666664 -e 0.001' # -g 可以选择核函数的参数，auto表示自动选择，scale表示1/(n_features * X.var())


# 训练模型
model = svm_train(y_train, x_train, param)

# 预测
p_labels, p_acc, p_vals = svm_predict(y_test, x_test, model)

# 准确率
accuracy = p_acc[0]  # p_acc[0] 返回的是准确率的小数形式
print(f'Accuracy: {accuracy}%')
print(f'Time: {time.time() - t0:.2f}s')