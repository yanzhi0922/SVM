from sklearn.datasets import load_svmlight_file
import numpy as np
from math import fabs
from scipy.sparse import csr_matrix
import multiprocessing as mp
import time
import os
# 读取数据
X_train, y_train = load_svmlight_file('madelon.txt')

X_test, y_test = load_svmlight_file('madelon.t')

epsilon = 0.001

'''
def cal_K(i, j):
    return X_train[i].dot(X_train[j].T)[0, 0]
'''
n_features = X_train.shape[1]
var_exp = X_train.data.var()  # 计算稀疏矩阵数据的方差
gamma = 1.0 / (n_features * var_exp)
def cal_K(i, j):
    diff = X_train[i] - X_train[j]
    return np.exp(-gamma * diff.dot(diff.T)[0, 0])

def compute_K_chunk(indices):
    K_chunk = np.zeros((len(indices), X_train.shape[0]))
    for idx, i in enumerate(indices):
        for j in range(X_train.shape[0]):
            K_chunk[idx, j] = cal_K(i, j)
    return K_chunk

def parallel_compute_K(num_processes):
    num_samples = X_train.shape[0]
    chunk_size = num_samples // num_processes
    indices = [range(i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes)]
    if num_samples % num_processes != 0:
        indices[-1] = range(indices[-1].start, num_samples)

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(compute_K_chunk, indices)

    K = np.vstack(results)
    return K



def cal_g(i, alpha, b):
    gi = b
    for j in range(X_train.shape[0]):
        gi += alpha[j] * y_train[j] * K[i, j]
    return gi

def KKT(i, alpha, C):
    gi = g[i]
    if alpha[i] < epsilon:
        return y_train[i] * gi >= 1
    elif alpha[i] > C - epsilon:
        return y_train[i] * gi <= 1
    else:
        return fabs(y_train[i] * gi - 1) < epsilon

def selectI(alpha, C):
    for i in range(X_train.shape[0]):
        if 0 < alpha[i] < C and not KKT(i, alpha, C):
            return i
    for i in range(X_train.shape[0]):
        if not KKT(i, alpha, C):
            return i
    return -1

def selectJ(Ei, E):
    if Ei > 0:
        j = np.argmin(E)
    else:
        j = np.argmax(E)
    return j


def SMO(C, max_iter=1000):
    alpha = np.zeros(X_train.shape[0])
    b = 0

    for i in range(X_train.shape[0]):
        g[i] = cal_g(i, alpha, b)

    for i in range(X_train.shape[0]):
        E[i] = g[i] - y_train[i]

    iter = 0
    while iter < max_iter:
        print('iter:', iter)
        # 外层循环，选择违反KKT条件的点
        i = selectI(alpha, C)
        if i == -1:
            break
        # 内层循环，选择使|Ei - Ej|最大的点
        j = selectJ(E[i], E)
        alpha_i_old = alpha[i]
        alpha_j_old = alpha[j]
        if y_train[i] != y_train[j]:
            L = max(0, alpha[j] - alpha[i])
            H = min(C, C + alpha[j] - alpha[i])
        else:
            L = max(0, alpha[j] + alpha[i] - C)
            H = min(C, alpha[j] + alpha[i])
        '''if L == H:
            continue'''
        eta = K[i, i] + K[j, j] - 2*K[i, j]
        while eta <= 0:
            j = np.random.randint(0, X_train.shape[0])
            eta = K[i, i] + K[j, j] - 2*K[i, j]

        alpha[j] = alpha_j_old + y_train[j]*(E[i]-E[j])/eta
        if alpha[j] > H:
            alpha[j] = H
        elif alpha[j] < L:
            alpha[j] = L
        alpha[i] = alpha_i_old + y_train[i]*y_train[j]*(alpha_j_old - alpha[j])
        # 更新Ei, Ej
        g[i] = cal_g(i, alpha, b)
        g[j] = cal_g(j, alpha, b)
        E[i] = g[i] - y_train[i]
        E[j] = g[j] - y_train[j]
        # 更新b
        b1 = - E[i] - y_train[i]*(alpha[i] - alpha_i_old)*K[i, i] - y_train[j]*(alpha[j] - alpha_j_old)*K[j, i] + b
        b2 = - E[j] - y_train[i]*(alpha[i] - alpha_i_old)*K[i, j] - y_train[j]*(alpha[j] - alpha_j_old)*K[j, j] + b
        b_old = b
        if 0 < alpha[i] < C:
            b = b1
        elif 0 < alpha[j] < C:
            b = b2
        else:
            b = (b1 + b2)/2
        for k in range(X_train.shape[0]):
                E[k] = E[k] - b_old + b
                g[k] = g[k] - b_old + b

        iter += 1
    return alpha, b



def predict(w, b):
    y_predict = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        if (np.dot(w.toarray().flatten(), X_test[i].toarray().flatten()) + b) > 0:
            y_predict[i] = 1
        else:
            y_predict[i] = -1
    return y_predict



def accuracy(y_test, y_predict):
    count = 0
    for i in range(len(y_test)):
        if y_test[i] == y_predict[i]:
            count += 1
    return count/len(y_test)



if __name__ == '__main__':
    t0 = time.time()
    mp.freeze_support()
    if 'K_rbf.txt' not in os.listdir():
        K = parallel_compute_K(mp.cpu_count())
        # 保存K到K.txt
        np.savetxt('K_rbf.txt', K)
    else:
        # 读取K
        K = np.loadtxt('K_rbf.txt')

    E = np.zeros(X_train.shape[0])
    g = np.zeros(X_train.shape[0])

    alpha, b = SMO(10, 1000)

    w = csr_matrix((X_train.shape[1], 1))
    for i in range(X_train.shape[0]):
        w += alpha[i] * y_train[i] * X_train[i].T

    y_predict = predict(w, b)
    print(accuracy(y_test, y_predict))
    print('time:', time.time() - t0)