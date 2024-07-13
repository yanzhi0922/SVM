from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
import numpy as np
# 读取数据
X_train, y_train = load_svmlight_file('madelon.txt')
X_test, y_test = load_svmlight_file('madelon.t')
epsilon = 0.001

param = {'kernel': 'sigmoid', 'C': 10.0, 'gamma': 1000, 'max_iter': 5000}
model = SVC(**param)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
print('accuracy:', accuracy)