from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
import numpy as np
# 读取数据
X_train, y_train = load_svmlight_file('madelon.txt')
X_test, y_test = load_svmlight_file('madelon.t')
epsilon = 0.001

from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
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