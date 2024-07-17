from libsvm.svmutil import *
import time
from sklearn.datasets import load_svmlight_file

t0 = time.time()
# 读取数据
X_train, Y_train = load_svmlight_file('splice.txt')
x_test, y_test = load_svmlight_file('splice.t')


#设置参数，使用RBF核

param = '-t 2 -c 10 -b 0 -g 0.016666666666666666 -e 0.001' # -g 可以选择核函数的参数，auto表示自动选择，scale表示1/(n_features * X.var())


# 训练模型
model = svm_train(Y_train, X_train, param)

# 预测
p_labels, p_acc, p_vals = svm_predict(y_test, x_test, model)

# 准确率
accuracy = p_acc[0]  # p_acc[0] 返回的是准确率的小数形式
print(f'Accuracy: {accuracy}%')
print(f'Time: {time.time() - t0:.2f}s')