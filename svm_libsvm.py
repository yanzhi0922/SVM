from libsvm.svmutil import *
import time

t0 = time.time()
# 读取数据
y_train, x_train = svm_read_problem('madelon.txt')
y_test, x_test = svm_read_problem('madelon.t')


# 设置参数，使用 RBF 核函数
param = '-t 2 -c 10 -b 0 -g 1.9794151456041003e-06'

# 训练模型
model = svm_train(y_train, x_train, param)

# 预测
p_labels, p_acc, p_vals = svm_predict(y_test, x_test, model)

# 计算准确率
accuracy = p_acc[0]  # p_acc[0] 返回的是准确率的小数形式
print(f'Accuracy: {accuracy}%')
print(f'Time: {time.time() - t0:.2f}s')