from libsvm.svmutil import *

# 读取数据
y_train, x_train = svm_read_problem('madelon.txt')
y_test, x_test = svm_read_problem('madelon.t')

# 设置参数
param = '-t 0 -c 1 -b 1'  # 线性核函数，C=1，概率输出

# 训练模型
model = svm_train(y_train, x_train, param)

# 预测
p_label, p_acc, p_val = svm_predict(y_test, x_test, model, '-b 1')

# 计算准确率
accuracy = p_acc[0]
print(f'Accuracy: {accuracy}%')