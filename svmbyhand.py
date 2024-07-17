import numpy as np
from sklearn.datasets import load_svmlight_file
import numpy as np
import time
def kernel(x, y):
    global gamma
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(- gamma*(np.linalg.norm(x - y, 2)) ** 2)
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(- gamma*(np.linalg.norm(x - y, 2, axis=1) ** 2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(- gamma*(np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2))
    return result


def takeStep(i1, i2):
    global alpha, b, E, C, eps, X_train, Y_train
    if i1 == i2:
        return 0

    alph1 = alpha[i1]
    y1 = Y_train[i1]
    E1 = E[i1]

    alph2 = alpha[i2]
    y2 = Y_train[i2]
    E2 = E[i2]

    s = y1 * y2

    if y1 != y2:
        L = max(0, alph2 - alph1)
        H = min(C, C + alph2 - alph1)
    else:
        L = max(0, alph2 + alph1 - C)
        H = min(C, alph2 + alph1)

    if L == H:
        return 0

    k11 = kernel(X_train[i1], X_train[i1])
    k12 = kernel(X_train[i1], X_train[i2])
    k22 = kernel(X_train[i2], X_train[i2])

    eta = k11 + k22 - 2 * k12

    if eta > 0:
        a2 = alph2 + (y2 * (E1 - E2)) / eta
        if a2 < L:
            a2 = L
        elif a2 > H:
            a2 = H
    else:
        alpha_tmp = alpha.copy()

        alpha_tmp[i2] = L
        L_obj = objective_function(alpha_tmp, Y_train, X_train)  # Objective function at a2=L

        alpha_tmp[i2] = H
        H_obj = objective_function(alpha_tmp, Y_train, X_train)  # Objective function at a2=H

        if L_obj < H_obj - eps:
            a2 = L
        elif L_obj > H_obj + eps:
            a2 = H
        else:
            a2 = alph2

    if a2 < eps:
        a2 = 0
    elif a2 > C-eps:
        a2 = C

    if abs(a2 - alph2) < eps * (a2 + alph2 + eps):
        return 0

    a1 = alph1 + s * (alph2 - a2)


    # 更新threshold
    b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + b
    b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + b

    if 0 < a1 < C:
        b_new = b1
    elif 0 < a2 < C:
        b_new = b2
    else:
        b_new = (b1 + b2) / 2.0

    # 更新alpha
    alpha[i1] = a1
    alpha[i2] = a2
    # 更新E
    for index, alph in zip([i1, i2], [a1, a2]):
        if 0.0 < alph < C:
            E[index] = 0.0

    NonOpt = list(filter(lambda n: n != i1 and n != i2, list(range(len(alpha)))))
    E[NonOpt] = (E[NonOpt] + y1 * (a1 - alph1) * kernel(X_train[i1], X_train[NonOpt]) +
                 y2 * (a2 - alph2) * kernel(X_train[i2], X_train[NonOpt]) + b - b_new)

    # 更新b
    b = b_new
    return 1


def examineExample(i2):
    global alpha, b, E, C, tol, X_train, Y_train
    y2 = Y_train[i2]
    alph2 = alpha[i2]
    E2 = E[i2]
    i1 = 0
    r2 = E2 * y2

    if (r2 < -tol and alph2 < C) or (r2 > tol and alph2 > 0):

        if len([i for i in range(len(alpha)) if 0 < alpha[i] < C]) > 1:
            # 暂定策略
            if E2 > 0:
                i1 = np.argmin(E)
            elif E2 <= 0:
                i1 = np.argmax(E)

            if takeStep(i1, i2):
                return 1

        for i1 in np.roll(np.where((alpha != 0) & (alpha != C))[0],np.random.choice(np.arange(len(alpha)))):
            if takeStep(i1, i2):
                return 1

        for i1 in np.roll(np.arange(len(alpha)), np.random.choice(np.arange(len(alpha)))):
            if takeStep(i1, i2):
                return 1

    return 0



def main_routine(y_train, x_train):
    numChanged = 0
    examineAll = 1

    while numChanged > 0 or examineAll:
        numChanged = 0
        if examineAll:
            for i in range(len(y_train)):
                numChanged += examineExample(i)
        else:
            for i in np.where((alpha != 0) & (alpha != C))[0]:
                if 0 < alpha[i] < C:
                    numChanged += examineExample(i)

        if examineAll:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1

def predict(x_test):
    result = np.zeros(x_test.shape[0])
    for i,test in enumerate(x_test):
        result[i] = decision_function(Y_train, X_train, test)
        if result[i] > 0:
            result[i] = 1
        else:
            result[i] = -1
    return result

def accuracy(y_test, y_predict):
    count = 0
    for i in range(len(y_test)):
        if y_test[i] == y_predict[i]:
            count += 1
    return count/len(y_test)

def objective_function(alph, y, x):
    return np.sum(alph) - 0.5 * np.sum((y[:, None] * y[None, :]) * kernel(x, x) * (alph[:, None] * alph[None, :]))

def decision_function(y, x_train, x_test):
    result = (alpha * y) @ kernel(x_train, x_test) - b
    return result



if __name__ == '__main__':
    t0 = time.time()
    np.random.seed(0)
    eps = 1e-7
    tol = 0.001
    b = 0
    C = 10

    X_train, Y_train = load_svmlight_file('splice.txt')
    X_train = X_train.toarray()

    X_test, Y_test = load_svmlight_file('splice.t')
    X_test = X_test.toarray()

    alpha = np.zeros(len(Y_train))
    gamma = 0.016666666666666666
    E = decision_function(Y_train, X_train, X_train) - Y_train

    main_routine(Y_train, X_train)
    y_predict = predict(X_test)
    print(accuracy(Y_test, y_predict))
    print(f'Time: {time.time() - t0:.2f}s')