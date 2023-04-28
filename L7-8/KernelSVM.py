import math
import numpy as np
from cvxopt import solvers, matrix
from func import CreatData, drawSupport
import matplotlib.pyplot as plt
from SVM import kernelSVM


def work(alpha_SV, X, x, Y, kenel):
    """

    :param alpha_SV: 支撑向量的alpha
    :param X:支撑向量的X
    :param x:
    :param Y:支撑向量的Y
    :return:
    """
    sum = 0
    for i in range(len(alpha_SV)):
        if kenel == 'four':
            # 四次项
            sum += alpha_SV[i] * Y[i] * math.pow(np.dot(X[:, i].T, x) + 1, 4)
        elif kenel == 'gao':
            # 高斯
            sum += alpha_SV[i] * Y[i] * np.exp(-1 * np.linalg.norm(X[:, i] - x) ** 2)
    return sum


def acc(alpha_SV, X, Y, testX, testY, b, kenel):
    """

    :param alpha_SV:
    :param X:
    :param Y:
    :param testX:
    :param testY:
    :param b:
    :return:
    """
    Y1 = np.array([])
    for i in range(len(testY)):
        if b + work(alpha_SV, X, testX[:, i], Y, kenel) > 0:
            Y1 = np.append(Y1, 1)
        else:
            Y1 = np.append(Y1, -1)
    num = 0
    for i in range(len(Y1)):
        if Y1[i] == testY[i]:
            num += 1
    return num / len(testY)


def draw(data, label, X_SV, Y_SV, b, alpha, kenel):
    linex = np.linspace(np.min(data[0, :]) - 1, np.max(data[0, :]) + 1, 100)
    liney = np.linspace(np.min(data[1, :]) - 1, np.max(data[1, :]) + 1, 100)
    X, Y = np.meshgrid(linex, liney)
    Z = func(X, Y, b, X_SV, Y_SV, alpha, kenel)
    plt.contourf(X, Y, Z)

    m, n = data.shape
    x1 = x2 = y1 = y2 = np.array([])
    for i in range(n):
        if label[i] == 1:
            x1 = np.append(x1, data[0][i])
            y1 = np.append(y1, data[1][i])
        elif label[i] == -1:
            x2 = np.append(x2, data[0][i])
            y2 = np.append(y2, data[1][i])
    ax = plt.subplot(111)
    ax.scatter(x1, y1, s=10, c='red', marker='s', label='+1 data')
    ax.scatter(x2, y2, s=10, c='green', label='-1 data')
    plt.legend(loc="upper left")


def func(x, y, b, X_SV, Y_SV, alpha, kenel):
    gsvm = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(y)):
            point = np.array([x[i, j], y[i, j]])
            for k in range(len(Y_SV)):
                if kenel == 'four':
                    gsvm[i, j] = gsvm[i, j] + alpha[k] * Y_SV[k] * math.pow(np.dot(X_SV[:, k].T, point) + 1, 4)
                elif kenel == 'gao':
                    # 高斯
                    gsvm[i, j] = gsvm[i, j] + alpha[k] * Y_SV[k] * np.exp(-1 * np.linalg.norm(X_SV[:, k] - point) ** 2)
            gsvm[i, j] = gsvm[i, j] + b
            gsvm[i, j] = np.sign(gsvm[i, j])
    return gsvm


def KernelTrain(kenel, mean1, mean2, title):
    cov = np.eye(2)
    traindata, trainlabel, testdata, testlabel = CreatData(mean1, mean2, cov, number=300)
    alpha = kernelSVM(traindata, trainlabel, kenel)
    alpha_SV = np.array([])
    X_SV = np.empty([2, 1])
    Y_SV = np.array([])
    for i in range(len(alpha)):
        if alpha[i] > 1e-6:
            alpha_SV = np.append(alpha_SV, alpha[i])
            X_SV = np.insert(X_SV, -1, traindata[:, i], axis=1)
            Y_SV = np.append(Y_SV, trainlabel[i])
    X_SV = X_SV[:, 0:-1]
    m = np.random.randint(0, len(alpha_SV))
    b = Y_SV[m] - work(alpha_SV, X_SV, X_SV[:, m], Y_SV, kenel)
    acc_train = acc(alpha_SV, X_SV, Y_SV, traindata, trainlabel, b, kenel)
    acc_test = acc(alpha_SV, X_SV, Y_SV, testdata, testlabel, b, kenel)
    print(acc_train, acc_test)
    plt.figure()
    draw(traindata, trainlabel, X_SV, Y_SV, b, alpha_SV, kenel)
    drawSupport(traindata, trainlabel, alpha)
    plt.title('train_' + title + kenel)
    # plt.savefig('./image/' + 'train_'+ title + kenel +'.png')
    plt.show()
    plt.figure()
    draw(testdata, testlabel, X_SV, Y_SV, b, alpha_SV, kenel)
    plt.title('test_' + title + kenel)
    # plt.savefig('./image/' + 'test_' + title + kenel + '.png')
    plt.show()


# 第一类情况:
# mean1 = np.array([-5, 0])
# mean2 = np.array([0, 5])
# KernelTrain('four', mean1, mean2, '[5,0]')
# KernelTrain('gao', mean1, mean2, '[5,0]')

# # 改变均值向量
#
mean3 = np.array([3, 0])
mean4 = np.array([0, 3])
# KernelTrain('four', mean3, mean4, '[3,0]')
KernelTrain('gao', mean3, mean4, '[3,0]')
