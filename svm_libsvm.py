from libsvm.svmutil import *

# 读取数据
y_train, x_train = svm_read_problem('madelon.txt')
y_test, x_test = svm_read_problem('madelon.t')

# 设置参数，使用RBF核函数，自动选择gamma（相当于scale选项）
param = '-t 2 -c 1 -b 1 -g 0'

# 训练模型
model = svm_train(y_train, x_train, param)

# 预测
p_labels, p_acc, p_vals = svm_predict(y_test, x_test, model)

# 计算准确率
accuracy = p_acc[0] * 100  # p_acc[0] 返回的是准确率的小数形式
print(f'Accuracy: {accuracy}%')