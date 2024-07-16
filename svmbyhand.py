from sklearn.datasets import load_svmlight_file
import numpy as np
from math import fabs
import time
import os
import random



def cal_K(xi, xj):
    global gamma
    return np.exp(-1*gamma * np.linalg.norm(xi - xj)**2)


def cal_g(i):
    global b
    global alpha
    gxi = b
    index = [i for i, alpha in enumerate(alpha) if alpha != 0]
    for j in index:
        gxi += alpha[j] * y_train[j] * K[j][i]
    return gxi

def cal_E(i):
    return cal_g(i) - y_train[i]

# KKT条件
def KKT(i,C):
    global alpha
    global epsilon
    gi = cal_g(i)
    if alpha[i] == 0 and y_train[i] * gi >= 1:
        return True
    elif 0 < alpha[i] < C and abs(y_train[i] * gi-1) < epsilon:
        return True
    elif alpha[i] == C and y_train[i] * gi <= 1:
        return True
    return False


# 内层循环，选择使|Ei - Ej|最大的点
def selectJ(E1, i):
    E2 = 0
    maxE1_E2 = -1
    maxIndex = -1
    nozeroE = [i for i, Ei in enumerate(E) if Ei != 0]
    # 对每个非零Ei的下标i进行遍历
    for j in nozeroE:
        # 如果是第一个变量的下标，跳过，因为第一个变量α1在前面已经确定
        if j == i:
            continue
        # 计算E2
        E2_tmp = cal_E(j)
        # 如果|E1-E2|大于目前最大值
        if fabs(E1 - E2_tmp) > maxE1_E2:
            # 更新最大值
            maxE1_E2 = fabs(E1 - E2_tmp)
            # 更新最大值E2
            E2 = E2_tmp
            # 更新最大值E2的索引j
            maxIndex = j
    # 如果列表中没有非0元素了（对应程序最开始运行时的情况）
    if maxIndex == -1:
        maxIndex = i
        while maxIndex == i:
            # 获得随机数，如果随机数与第一个变量的下标i一致则重新随机
            maxIndex = int(random.uniform(0, X_train.shape[0]))
        # 获得E2
        E2 = cal_E(maxIndex)

    # 返回第二个变量的E2值以及其索引
    return E2, maxIndex


def SMO(C, max_iter=1000):
    # 初始化
    iter = 0
    global S
    flag = True
    global b
    global alpha
    while flag:
        print('iter:', iter+1)
        flag = False
        # 外层循环，选择违反KKT条件的点
        for i in range(X_train.shape[0]):
            if KKT(i, C)== False:
                Ei = cal_E(i)
                flag = True
                # 内层循环，选择使|Ei - Ej|最大的点
                Ej, j = selectJ(Ei, i)

                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]
                if y_train[i] != y_train[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                elif y_train[i] == y_train[j]:
                    L = max(0, alpha[j] + alpha[i] - C)
                    H = min(C, alpha[j] + alpha[i])

                if L == H:
                    continue

                eta = K[i][i] + K[j][j] - 2 * K[i][j]
                if eta <= 0:
                    continue

                alpha[j] = alpha_j_old + y_train[j] * (E[i] - E[j]) / eta
                # 裁剪
                if alpha[j] > H:
                    alpha[j] = H
                elif alpha[j] < L:
                    alpha[j] = L
                alpha[i] = alpha_i_old + y_train[i] * y_train[j] * (alpha_j_old - alpha[j])

                # 更新b
                b1 =b - E[i] - y_train[i] * (alpha[i] - alpha_i_old) * K[i][i] - y_train[j] * (alpha[j] - alpha_j_old) * K[j][i]
                b2 =b - E[j] - y_train[i] * (alpha[i] - alpha_i_old) * K[i][j] - y_train[j] * (alpha[j] - alpha_j_old) * K[j][j]
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                # 更新E
                E[i] = cal_E(i)
                E[j] = cal_E(j)
        iter += 1
    for i in range(X_train.shape[0]):
        if 0 < alpha[i]:
            S.append(i)
    return alpha, b, S



def predict(alpha, b):
    global S
    y_predict = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        tmp = b
        for j in S:
            tmp += alpha[j] * y_train[j] * K2[i][j]
        y_predict[i] = np.sign(tmp)
    return y_predict


def accuracy(y_test, y_predict):
    count = 0
    for i in range(len(y_test)):
        if y_test[i] == y_predict[i]:
            count += 1
    return count/len(y_test)



if __name__ == '__main__':
    # 读取数据
    X, Y = load_svmlight_file('german.numer_scale.txt')
    X = X.toarray()
    # 划分70%做训练集
    X_train = X[:int(0.7 * X.shape[0])]
    y_train = Y[:int(0.7 * Y.shape[0])]

    # 30%做测试集
    X_test = X[int(0.7 * X.shape[0]):]
    y_test = Y[int(0.7 * Y.shape[0]):]

    epsilon = 0.001
    t0 = time.time()
    gamma = 1 / X_train.shape[1]
    alpha = np.zeros(X_train.shape[0])
    # 初始化b为全局变量
    b = 0
    S = []

    K = np.zeros((X_train.shape[0], X_train.shape[0]))
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[0]):
            K[i][j] = cal_K(X_train[i], X_train[j])

    K2 = np.zeros((X_test.shape[0], X_train.shape[0]))
    for i in range(X_test.shape[0]):
        for j in range(X_train.shape[0]):
            K2[i][j] = cal_K(X_test[i], X_train[j])

    E = np.zeros(X_train.shape[0])

    alpha, b, S = SMO(10, 20000)

    y_predict = predict(alpha, b)
    print(accuracy(y_test, y_predict))
    print('time:', time.time() - t0)