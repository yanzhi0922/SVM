from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
import numpy as np
import time
# 读取数据
X, Y = load_svmlight_file('german.numer_scale.txt')
X = X.toarray()
#划分70%做训练集
X_train = X[:int(0.7 * X.shape[0])]
y_train = Y[:int(0.7 * Y.shape[0])]

# 30%做测试集
X_test = X[int(0.7 * X.shape[0]):]
y_test = Y[int(0.7 * Y.shape[0]):]
epsilon = 0.001
t0 = time.time()
from sklearn.model_selection import GridSearchCV
'''
# 定义参数网格
param_grid = {
    'C': [10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf']
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(SVC(max_iter=1000), param_grid, cv=5)

# 训练模型
grid_search.fit(X_train, y_train)

# 打印最佳参数和最佳分数
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# 使用最佳参数的模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 计算准确率
accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
print('accuracy:', accuracy)
'''
model = SVC(C=10, gamma=1/X_train.shape[1], kernel='rbf', max_iter=4000) # gamma = scale,表示
gamma = 1 / X_train.shape[1]
print(gamma)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
print('accuracy:', accuracy)
print(f'Time: {time.time() - t0:.2f}s')