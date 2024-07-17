from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
import numpy as np
import time
# 读取数据

X_train, Y_train = load_svmlight_file('splice.txt')
X_train = X_train.toarray()

X_test, y_test = load_svmlight_file('splice.t')
X_test = X_test.toarray()

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
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
print('accuracy:', accuracy)
print(f'Time: {time.time() - t0:.2f}s')