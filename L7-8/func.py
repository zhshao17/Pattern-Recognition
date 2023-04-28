import numpy as np
import matplotlib.pyplot as plt


def CreatData(mean1, mean2, cov, number):
    """

        :param mean1:
        :param mean2:
        :param cov:
        :param number:
        :return:
        """
    data1 = np.transpose(np.random.multivariate_normal(mean1, cov, number))
    data2 = np.transpose(np.random.multivariate_normal(mean2, cov, number))

    traindata = np.hstack([data1[:, 0:int(0.8 * number)], data2[:, 0:int(0.8 * number)]])
    trainlabel = np.hstack([np.ones(int(0.8 * number)), -1 * np.ones(int(0.8 * number))])
    testdata = np.hstack([data1[:, int(0.8 * number):], data2[:, int(0.8 * number):]])
    testlabel = np.hstack([np.ones(int(0.2 * number)), -1 * np.ones(int(0.2 * number))])

    # 数据增广化
    # traindata = np.vstack([np.ones(int(0.8 * number * 2)), traindata])
    # testdata = np.vstack([np.ones(int(0.2 * number * 2)), testdata])
    return traindata, trainlabel, testdata, testlabel


def acc(W, b, X, Y):
    """

    :param W:
    :param b:
    :param X:
    :param Y:
    :return:
    """
    m, n = X.shape
    Y1 = np.array([])
    for i in range(n):
        if np.dot(W, X[:, i]) + b > 0:
            Y1 = np.append(Y1, 1)
        elif np.dot(W, X[:, i]) + b <= 0:
            Y1 = np.append(Y1, -1)
    num = 0
    for i in range(n):
        if Y1[i] == Y[i]:
            num += 1
    return num / n


def draw(X, Y, w, b, title):
    m, n = X.shape
    x1 = x2 = y1 = y2 = np.array([])
    for i in range(n):
        if Y[i] == 1:
            x1 = np.append(x1, X[0][i])
            y1 = np.append(y1, X[1][i])
        elif Y[i] == -1:
            x2 = np.append(x2, X[0][i])
            y2 = np.append(y2, X[1][i])
    ax = plt.subplot(111)
    ax.scatter(x1, y1, s=10, c='red', marker='s', label='+1 data')
    ax.scatter(x2, y2, s=10, c='green', label='-1 data')
    a, d = np.amax(X[0, :]), np.amin(X[0, :])
    x = np.arange(d, a, 0.1)
    y = (-1 * b - w[0] * x) / w[1]
    delta_b = 1/w[1]
    y1 = (-1 * b - w[0] * x) / w[1] - delta_b  # 间隔面
    y2 = (-1 * b - w[0] * x) / w[1] + delta_b  # 间隔面
    ax.plot(x, y, label='Classification surface')
    ax.plot(x, y1, label='spacer plane1')
    ax.plot(x, y2, label='spacer plane2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)


def drawSupport(X, Y, alpha):
    for i in range(len(Y)):
        if alpha[i] > 1e-6:
            if Y[i] == 1:
                plt.scatter(X[0, i], X[1, i], s=10, c='black', marker='s')
            elif Y[i] == -1:
                plt.scatter(X[0, i], X[1, i], s=10, c='black')
    plt.legend(loc="upper left")


def drawdiaoyu(X, Y, w, b, title):
    m, n = X.shape
    x1 = x2 = y1 = y2 = np.array([])
    for i in range(n):
        if Y[i] == 1:
            x1 = np.append(x1, X[0][i])
            y1 = np.append(y1, X[1][i])
        elif Y[i] == -1:
            x2 = np.append(x2, X[0][i])
            y2 = np.append(y2, X[1][i])
    ax = plt.subplot(111)
    ax.scatter(x1, y1, s=10, c='red', marker='s', label='china')
    ax.scatter(x2, y2, s=10, c='green', label='japan')
    a, d = np.amax(X[0, :]), np.amin(X[0, :])
    x = np.arange(d, a, 0.1)
    y = (-1 * b - w[0] * x) / w[1]
    delta_b = 1 / w[1]
    y1 = (-1 * b - w[0] * x) / w[1] - delta_b  # 间隔面
    y2 = (-1 * b - w[0] * x) / w[1] + delta_b  # 间隔面
    ax.plot(x, y, label='Classification surface')
    ax.plot(x, y1, label='spacer plane1')
    ax.plot(x, y2, label='spacer plane2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)