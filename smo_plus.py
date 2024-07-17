import numpy as np


# k(x1, x2) = <x1, x2>
def linear_kernel(x1, x2):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    return x1.dot(x2.T)


# k(x, y) = exp(- gamma ||x1 - x2||^2)
def get_rbf_kernel(gamma):
    def rbf_kernel(x1, x2):
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        s1, _ = x1.shape
        s2, _ = x2.shape
        norm1 = np.ones((s2, 1)).dot(np.atleast_2d(np.sum(x1 ** 2, axis=1))).T
        norm2 = np.ones((s1, 1)).dot(np.atleast_2d(np.sum(x2 ** 2, axis=1)))
        return np.exp(- gamma * (norm1 + norm2 - 2 * x1.dot(x2.T)))
    return rbf_kernel

class SVM:

    def __init__(self, C=1.0, kernel='rbf', gamma='auto', coef0=0.0, tol=1e-3, max_iter=-1):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.eps = 1e-7
        self.kernel_func = None
        self.alpha = None
        self.Ecache = None
        self.i_low, self.i_up = None, None
        self.b_low, self.b_up = None, None
        self.I_0, self.I_1, self.I_2, self.I_3, self.I_4 = None, None, None, None, None
        self.coef = None
        self.dual_coef = None
        self.threshold = None
        self.support = None
        self.support_vectors = None
        self.bounded = None

    def get_gamma(self, X):
        if isinstance(self.gamma, float):
            return self.gamma
        elif self.gamma == 'auto':
            return 1.0 / X.shape[1]
        elif self.gamma == 'scale':
            X_var = X.var()
            return 1.0 / (X.shape[1] * X_var) if X_var > self.eps else 1.0
        else:
            raise ValueError(f"'{self.gamma}' is incorrect value for gamma")

    def get_kernel_function(self, X):
        if callable(self.kernel):
            return self.kernel
        elif self.kernel == 'linear':
            return linear_kernel
        elif self.kernel == 'rbf':
            return get_rbf_kernel(self.get_gamma(X))
        else:
            raise ValueError(f"'{self.kernel}' is incorrect value for kernel")

    def compute_L_H(self, y1, y2, alpha1, alpha2):
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)
        return L, H

    def compute_objective_function(self, y1, y2, F1, F2, alpha1, alpha2, s, k11, k12, k22, L, H):
        f1 = y1 * F1 - alpha1 * k11 - s * alpha2 * k12
        f2 = y2 * F2 - s * alpha1 * k12 - alpha2 * k22
        L1 = alpha1 + s * (alpha2 - L)
        H1 = alpha1 + s * (alpha2 - H)
        Psi_L = L1 * f1 + L * f2 + 0.5 * L1 ** 2 * k11 + 0.5 * L ** 2 * k22 + s * L * L1 * k12
        Psi_H = H1 * f1 + H * f2 + 0.5 * H1 ** 2 * k11 + 0.5 * H ** 2 * k22 + s * H * H1 * k12
        return Psi_L, Psi_H

    def update_I(self, i, y, a):
        if self.I_0[i]:
            self.I_0[i] = False
        else:
            if y == 1:
                if self.I_1[i]:
                    self.I_1[i] = False
                else:
                    self.I_3[i] = False
            else:
                if self.I_2[i]:
                    self.I_2[i] = False
                else:
                    self.I_4[i] = False
        if a <= self.eps or a >= self.C - self.eps:
            if y == 1:
                if a <= self.eps:
                    self.I_1[i] = True
                else:
                    self.I_3[i] = True
            else:
                if a <= self.eps:
                    self.I_4[i] = True
                else:
                    self.I_2[i] = True
        else:
            self.I_0[i] = True

    def update_I_low_up(self, I_low, I_up, i):
        if self.I_3[i] or self.I_4[i]:
            I_low[i] = True
        else:
            I_up[i] = True

    def get_b_i(self, I, argfunc):
        I = np.where(I)[0]
        E = self.Ecache[I]
        i = I[argfunc(E)]
        b = self.Ecache[i]
        return b, i

    def take_step(self, i, j, X, y):
        if i == j:
            return False
        y1 = y[i]
        y2 = y[j]
        alpha1 = self.alpha[i]
        alpha2 = self.alpha[j]
        E1 = self.Ecache[i]
        E2 = self.Ecache[j]
        s = y1 * y2
        L, H = self.compute_L_H(y1, y2, alpha1, alpha2)  # Compute L and H
        if abs(L - H) < self.eps:
            return False
        k11 = self.kernel_func(X[i], X[i])[0, 0]
        k12 = self.kernel_func(X[i], X[j])[0, 0]
        k22 = self.kernel_func(X[j], X[j])[0, 0]
        eta = 2 * k12 - k11 - k22
        if eta < -self.eps:
            a2 = alpha2 - y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            # Compute objective function at a2 = L and a2 = H
            L_obj, H_obj = self.compute_objective_function(y1, y2, E1, E2, alpha1, alpha2, s, k11, k12, k22, L, H)
            if L_obj < H_obj - self.eps:
                a2 = L
            elif L_obj > H_obj + self.eps:
                a2 = H
            else:
                a2 = alpha2
        if abs(a2 - alpha2) < self.eps * (a2 + alpha2 + self.eps):
            return False
        a1 = alpha1 + s * (alpha2 - a2)
        # Update Ecache[i] for i in I_0 using new Lagrange multipliers
        ki1 = self.kernel_func(X[self.I_0], X[i]).ravel()
        ki2 = self.kernel_func(X[self.I_0], X[j]).ravel()
        self.Ecache[self.I_0] += y1 * (a1 - alpha1) * ki1 + y2 * (a2 - alpha2) * ki2
        # Store a1 and a2 in the alpha array
        self.alpha[i] = a1
        self.alpha[j] = a2
        # Update I_0, I_1, I_2, I_3 and I_4
        self.update_I(i, y1, a1)
        self.update_I(j, y2, a2)
        # Compute updated E values for i1 and i2
        self.Ecache[i] = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12
        self.Ecache[j] = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22
        # Compute (i_low, b_low) and (i_up, b_up)
        I_low, I_up = self.I_0.copy(), self.I_0.copy()
        self.update_I_low_up(I_low, I_up, i)
        self.update_I_low_up(I_low, I_up, j)
        self.b_low, self.i_low = self.get_b_i(I_low, np.argmax)
        self.b_up, self.i_up = self.get_b_i(I_up, np.argmin)
        return True

    def compute_Ei(self, X, y, i):
        return np.sum(self.kernel_func(X[i], X).ravel() * (self.alpha * y)) - y[i]

    def examine_example(self, i2, X, y):
        if self.I_0[i2]:
            E2 = self.Ecache[i2]
        else:
            E2 = self.compute_Ei(X, y, i2)  # compute F_i2
            self.Ecache[i2] = E2
            # Update (b_low, i_low) or (b_up, i_up) using (E2, i2)
            if (self.I_1[i2] or self.I_2[i2]) and E2 < self.b_up:
                self.b_up, self.i_up = E2, i2
            elif (self.I_3[i2] or self.I_4[i2]) and E2 > self.b_low:
                self.b_low, self.i_low = E2, i2
        # Check optimality using current b_low and b_up
        # If violated, find an index i1 to do joint optimization with i2
        optimality = True
        i1 = 0
        if (self.I_0[i2] or self.I_1[i2] or self.I_2[i2]) and self.b_low - E2 > 2 * self.tol:
            optimality = False
            i1 = self.i_low
        if (self.I_0[i2] or self.I_3[i2] or self.I_4[i2]) and E2 - self.b_up > 2 * self.tol:
            optimality = False
            i1 = self.i_up
        if optimality:
            return 0
        # For i2 in I_0 choose the better i1
        if self.I_0[i2]:
            if self.b_low - E2 > E2 - self.b_up:
                i1 = self.i_low
            else:
                i1 = self.i_up
        return int(self.take_step(i1, i2, X, y))

    def initialize_fitting(self, X, y):
        X, y = np.asarray(X), np.asarray(y)

        self.alpha = np.zeros(y.shape[0])
        self.Ecache = np.zeros(y.shape[0])

        self.b_up = -1
        y1 = y == 1
        self.i_up = y1.nonzero()[0][0]
        self.I_1 = y1

        self.b_low = 1
        y2 = y == -1
        self.i_low = y2.nonzero()[0][0]
        self.I_4 = y2

        self.Ecache[self.i_low] = 1
        self.Ecache[self.i_up] = -1

        self.I_0, self.I_2, self.I_3 = np.zeros(y.shape, bool), np.zeros(y.shape, bool), np.zeros(y.shape, bool)

        self.kernel_func = self.get_kernel_function(X)
        max_iter = self.max_iter if self.max_iter >= 0 else np.inf

        return X, y, max_iter

    def set_result(self, X, y):
        self.support = np.where(self.alpha > self.eps)[0]  # 将alpha大于0的点作为支持向量
        self.support_vectors = X[self.support] # 支持向量
        self.bounded = np.where(self.alpha >= self.C - self.eps)[0]  # 边界点
        self.dual_coef = self.alpha[self.support] * y[self.support]
        self.threshold = (self.b_low + self.b_up) / 2
        if self.kernel == 'linear':
            self.coef = np.atleast_2d(np.sum(self.dual_coef * self.support_vectors.T, axis=1))

    def fit_modification(self, X, y):
        X, y, max_iter = self.initialize_fitting(X, y)
        iteration = 0
        num_changed = 0
        examine_all = True
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(y.shape[0]):
                    num_changed += self.examine_example(i, X, y)
            else:
                # Loop over all examples where alpha is not 0 & not C
                for i in np.where(self.I_0)[0]:
                    num_changed += self.examine_example(i, X, y)
                    if self.b_up > self.b_low - 2 * self.tol:
                        num_changed = 0
                        break
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            iteration += 1
            if iteration >= max_iter:
                break
        self.set_result(X, y)