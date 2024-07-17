import time
import numpy as np
import math
import random
from sklearn.datasets import load_svmlight_file

class SVM:
    """
    SVM类
    """

    def __init__(self, trainDataList, trainLabelList, gamma=10.0, C=20000.0, epsilon=0.0, maxIter=100):
        """
        SVM相关参数初始化
        :param trainDataList:训练数据集
        :param trainLabelList: 训练测试集
        :param gamma: RBF核函数中的γ
        :param C:软间隔中的惩罚参数 对于硬间隔支持向量机，设置为一个较大的数
        :param epsilon:松弛变量 对于硬间隔支持向量机 设置为0
        """
        self.trainDataMat = np.asmatrix(trainDataList)  # 训练数据集
        self.y_train = np.asmatrix(trainLabelList).T  # 训练标签集，为了方便后续运算提前做了转置，变为列向量
        self.m, self.n = np.shape(self.trainDataMat)  # m：训练集数量    n：样本特征数目
        self.gamma = gamma  # 核分母中的σ，建议先使用高斯核函数,之后可以尝试其它核函数
        self.C = C  # 惩罚参数（与软间隔支持向量机相关）
        self.epsilon = epsilon  # 松弛变量，为0即硬间隔支持向量机
        self.k = self.calcKernel()  # 核函数对应的gram矩阵（初始化时提前计算）
        self.b = 0  # SVM中的偏置b
        self.b_low = None
        self.b_up = None
        self.alpha = [0] * self.trainDataMat.shape[0]  # α 拉格朗日因子，其长度为训练集数目
        self.E = [0 * self.y_train[i, 0] for i in range(self.y_train.shape[0])]  # SMO运算过程中的Ei
        self.supportVecIndex = []  # 支持向量的对应索引
        self.maxIter = maxIter  # 最大迭代次数
    def calcKernel(self):
        """
        计算核函数，推荐使用高斯核
        :return: 核矩阵
        """
        # 初始化核结果矩阵， 其大小 = 训练集长度m * 训练集长度m

        # k[i][j] = Xi * Xj，X为训练或测试数据
        # 书中7.90式中的k(xi, xj) = φ(xi) * φ(xj)
        k = [[0 for i in range(self.m)] for j in range(self.m)]
        for i in range(self.m):
            xi = self.trainDataMat[i, :]
            for j in range(self.m):
                xj = self.trainDataMat[j, :]
                k[i][j] = self.calcSinglKernel(xi, xj)
        # 返回核矩阵
        return k

    def isSatisfyKKT(self, i):
        """
        查看第i个α是否满足KKT条件
        :param i:α的下标
        :return:
            True：满足
            False：不满足
        """
        Ei = self.calcEi(i)
        r = self.y_train[i] * Ei
        if (r < -self.epsilon and self.alpha[i] < self.C) or (r > self.epsilon and self.alpha[i] > 0):
            return True
        return False

    def calc_gxi(self, i):
        """
        计算g(xi)，未进行符号函数离散的目标函数值
        :param i:x的下标
        :return: g(xi)的值
        """
        # 提示
        # g(xi)为求和式，是否可以根据支持向量的相关性质，简化求和计算
        # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        # 例如：
        # list1 = ['a', 'b', 'c', 'd']
        # for index, item in enumerate(list1):
        #     print(index, '-', item)
        # 输出：
        # 0 - a
        # 1 - b
        # 2 - c
        # 3 - d
        gxi = 0
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        for j in index:
            gxi += self.alpha[j] * self.y_train[j] * self.k[j][i]
        gxi -= self.b  # 返回g(xi)
        return gxi

    def calcEi(self, i):
        """
        计算Ei
        :param i: E的下标
        :return:
        """
        # 计算g(xi)
        gxi = self.calc_gxi(i)
        return gxi - self.y_train[i]

    def getAlphaJ(self, E1, i):
        """
        SMO中选择第二个变量
        :param E1: 第一个变量的E1
        :param i: 第一个变量α的下标
        :return: E2，α2的下标
        """
        # 初始化E2
        E2 = 0
        # 初始化|E1-E2|为-1
        maxE1_E2 = -1
        # 初始化第二个变量的下标
        maxIndex = -1

        # 这一步是一个优化性的算法
        # 实际上书上算法中初始时每一个Ei应当都为-yi（因为g(xi)由于初始α为0，必然为0）
        # 然后每次按照书中第二步去计算不同的E2来使得|E1-E2|最大，但是时间耗费太长了
        # 作者最初是全部按照书中缩写，但是本函数在需要3秒左右，所以进行了一些优化措施
        # --------------------------------------------------
        # 在Ei的初始化中，由于所有α为0，所以一开始是设置Ei初始值为-yi。这里修改为与α一致，初始状态所有Ei为0，在运行过程中再逐步更新
        # 因此在挑选第二个变量时，只考虑更新过Ei的变量，但是存在问题
        # 1.当程序刚开始运行时，所有Ei都是0，那挑谁呢？
        #   当程序检测到并没有Ei为非0时，将会使用随机函数随机挑选一个
        # 2.怎么保证能和书中的方法保持一样的有效性呢？
        #   在挑选第一个变量时是有一个大循环的，它能保证遍历到每一个xi，并更新xi的值，
        # 在程序运行后期后其实绝大部分Ei都已经更新完毕了。下方优化算法只不过是在程序运行
        # 的前半程进行了时间的加速，在程序后期其实与未优化的情况无异
        # ------------------------------------------------------

        # 获得Ei非0的对应索引组成的列表，列表内容为非0Ei的下标i
        E_tmp = [i for i, Ei in enumerate(self.E) if Ei != 0]
        # 对每个非零Ei的下标i进行遍历
        for j in E_tmp:
            """
            第二个变量a的选取，需要同学们补充
            即如何进行内循环，进而更新maxIndex
            """
            # 如果是第一个变量的下标，跳过，因为第一个变量α1在前面已经确定
            if j == i:
                continue
            # 计算E2
            E2_tmp = self.calcEi(j)
            # 如果|E1-E2|大于目前最大值
            if np.fabs(E1 - E2_tmp) > maxE1_E2:
                # 更新最大值
                maxE1_E2 = np.fabs(E1 - E2_tmp)
                # 更新最大值E2
                E2 = E2_tmp
                # 更新最大值E2的索引j
                maxIndex = j
        # 如果列表中没有非0元素了（对应程序最开始运行时的情况）
        if maxIndex == -1:
            maxIndex = i
            while maxIndex == i:
                # 获得随机数，如果随机数与第一个变量的下标i一致则重新随机
                maxIndex = int(random.uniform(0, self.m))
            # 获得E2
            E2 = self.calcEi(maxIndex)

        # 返回第二个变量的E2值以及其索引
        return E2, maxIndex

    def train(self):
        # iterStep：迭代次数，超过设置次数还未收敛则强制停止
        # parameterChanged：单次迭代中有参数改变则增加1
        iterStep = 0
        parameterChanged = 1

        #  如果没有达到限制的迭代次数以及上次迭代中有参数改变则继续迭代
        # parameterChanged==0时表示上次迭代没有参数改变，如果遍历了一遍都没有参数改变，说明
        # 达到了收敛状态，可以停止了
        examine_all = True
        while parameterChanged > 0:
            # 打印当前迭代轮数
            print('iter:%d:%d' % (iterStep, self.maxIter))
            # 迭代步数加1
            iterStep += 1
            # 新的一轮将参数改变标志位重新置0
            parameterChanged = 0
            # 大循环遍历所有样本，用于找SMO中第一个变量
            for i in range(self.m):
                # 查看第一个遍历是否满足KKT条件，如果不满足则作为SMO中第一个变量从而进行优化
                if self.isSatisfyKKT(i) == False:
                    # 如果下标为i的α不满足KKT条件，则进行优化
                    flag = True
                    # 第一个变量α的下标i已经确定，接下来按照变量的选择方法第二步
                    # 选择变量2。由于变量2的选择中涉及到|E1 - E2|，因此先计算E1
                    E1 = self.calcEi(i)

                    # 选择第2个变量
                    E2, j = self.getAlphaJ(E1, i)

                    # 获得两个变量的标签
                    y1 = self.y_train[i]
                    y2 = self.y_train[j]
                    # 复制α值作为old值
                    alphaOld_1 = self.alpha[i]
                    alphaOld_2 = self.alpha[j]
                    # 依据标签是否一致来生成不同的L和H
                    if y1 != y2:
                        L = max(0, alphaOld_2 - alphaOld_1)
                        H = min(self.C, self.C + alphaOld_2 - alphaOld_1)
                    else:
                        L = max(0, alphaOld_2 + alphaOld_1 - self.C)
                        H = min(self.C, alphaOld_2 + alphaOld_1)
                    # 如果两者相等，说明该变量无法再优化，直接跳到下一次循环
                    if L == H:
                        continue
                    # 先获得几个k值，用来计算7.106中的分母η
                    k11 = self.k[i][i]
                    k22 = self.k[j][j]
                    k21 = self.k[j][i]
                    k12 = self.k[i][j]
                    eta = k11 + k22 - 2 * k12
                    if eta > 0:
                        # 更新α2，该α2还未经剪切
                        alphaNew_2 = alphaOld_2 + y2 * (E1 - E2) / eta
                        # 剪切α2
                        if alphaNew_2 < L:
                            alphaNew_2 = L
                        elif alphaNew_2 > H:
                            alphaNew_2 = H
                    else :
                        Lobj = self.compute_obj(L, i, j)
                        Hobj = self.compute_obj(H, i, j)
                        if Lobj < Hobj - 0.00001:
                            alphaNew_2 = L
                        elif Lobj > Hobj + 0.00001:
                            alphaNew_2 = H
                        else:
                            alphaNew_2 = alphaOld_2


                    # 更新α1
                    alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2)

                    # 计算b1和b2
                    b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k21 * (alphaNew_2 - alphaOld_2) + self.b
                    b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k22 * (alphaNew_2 - alphaOld_2) + self.b

                    # 依据α1和α2的值范围确定新b
                    if (alphaNew_1 > 0) and (alphaNew_1 < self.C):
                        bNew = b1New
                    elif (alphaNew_2 > 0) and (alphaNew_2 < self.C):
                        bNew = b2New
                    else:
                        bNew = (b1New + b2New) / 2

                    # 将更新后的各类值写入，进行更新
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.b = bNew

                    self.E[i] = self.calcEi(i)
                    self.E[j] = self.calcEi(j)

                    # 如果α2的改变量过于小，就认为该参数未改变，不增加parameterChanged值
                    # 反之则自增1
                    if np.abs(alphaNew_2 - alphaOld_2) >= self.epsilon*(alphaNew_2 + alphaOld_2 + self.epsilon):
                        parameterChanged += 1

                # 打印迭代轮数，i值，该迭代轮数修改α数目
                print("iter: %d i:%d, pairs changed %d" % (iterStep, i, parameterChanged))

        # 全部计算结束后，重新遍历一遍α，查找里面的支持向量
        for i in range(self.m):
            # 如果α>0，说明是支持向量
            if self.alpha[i] > 0:
                # 将支持向量的索引保存起来
                self.supportVecIndex.append(i)

    def calcSinglKernel(self, x1, x2):
        """
        单独计算核函数
        :param x1:向量1
        :param x2: 向量2
        :return: 核函数结果
        """
        # RBF核函数
        result = np.exp(-1*self.gamma * np.linalg.norm(x1 - x2))
        # 返回结果
        return result

    def predict(self, x):
        """
        对样本的标签进行预测
        :param x: 要预测的样本x
        :return: 预测结果
        """
        result = 0
        for i in self.supportVecIndex:
            # 遍历所有支持向量，计算求和式
            # 如果是非支持向量，求和子式必为0，没有必须进行计算
            # 这也是为什么在SVM最后只有支持向量起作用
            # ------------------
            # 先单独将核函数计算出来
            tmp = self.calcSinglKernel(self.trainDataMat[i, :], np.asmatrix(x))
            # 对每一项子式进行求和，最终计算得到求和项的值
            result += self.alpha[i] * self.y_train[i] * tmp
        # 求和项计算结束后加上偏置b
        result -= self.b
        # 使用sign函数返回预测结果
        return np.sign(result)

    def test(self, testDataList, testLabelList):
        """
        测试
        :param testDataList:测试数据集
        :param testLabelList: 测试标签集
        :return: 正确率
        """
        # 错误计数值
        errorCnt = 0
        # 遍历测试集所有样本
        for i in range(len(testDataList)):
            # 获取预测结果
            result = self.predict(testDataList[i])
            # 如果预测与标签不一致，错误计数值加一
            if result != testLabelList[i]:
                errorCnt += 1
        # 返回正确率
        return 1 - errorCnt / len(testDataList)


if __name__ == '__main__':
    start = time.time()

    # 获取数据集及标签
    print('start read DataSet')
    X, Y = load_svmlight_file('heart.txt')
    X = X.toarray()
    #划分70%做训练集
    X_train = X[:int(0.7 * X.shape[0])]
    Y_train = Y[:int(0.7 * Y.shape[0])]

    # 30%做测试集
    X_test = X[int(0.7 * X.shape[0]):]
    Y_test = Y[int(0.7 * Y.shape[0]):]

    # 初始化SVM类
    print('start init SVM')
    # 通过实际实验，修改各参数的值，以达到最好的实验效果
    svm = SVM(X_train, Y_train, 1 / X_train.shape[1], 10, 0.001,3000)

    # 开始训练
    print('start to train')
    svm.train()

    # 开始测试
    print('start to test')
    accuracy = svm.test(X_test, Y_test)
    print('the accuracy is:%d' % (accuracy * 100), '%')

    # 打印时间
    print('time span:', time.time() - start)
