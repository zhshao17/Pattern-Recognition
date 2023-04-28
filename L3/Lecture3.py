import numpy as np
import math
import matplotlib.pyplot as plt
import time


def Gatter(X, Y):
    """

    :param X: 默认X已经增广化， X:3*320
    :param Y: 列向量
    :return:
    """
    # X的广义逆
    X1 = np.dot(np.linalg.inv((np.dot(X, np.transpose(X)))), X)
    # W^*
    W = np.dot(X1, Y)
    return W


def Grad(X, Y, learning_rate, epochs):
    """

    :param epochs:
    :param learning_rate:
    :param X:
    :param Y:
    :return:
    """
    m, n = X.shape
    # 初始化权重
    W0 = np.random.random(m)
    Wt = W0
    k = 0
    loss_record = np.array([])
    for k in range(epochs):
        loss = 0
        grad = np.zeros(m)
        for i in range(n):
            loss += (np.dot(Wt, X[:, i]) - Y[i]) ** 2
            grad += (np.dot(Wt, X[:, i]) - Y[i]) * X[:, i]
        loss_record = np.append(loss_record, loss / n)
        Wt = Wt - learning_rate * (2 * grad / n)
        if np.linalg.norm(grad) < 1e-10:
            break
    x = np.arange(k + 1)
    plt.plot(x, loss_record)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    return Wt


def random(X, Y):
    m, n = X.shape
    indices = np.random.permutation(n)
    rand_X = X[:, indices]
    rand_Y = Y[indices]
    return rand_X, rand_Y


def SGD(X, Y, learning_rate, epochs, batch_size):
    """

    :param batch_size:
    :param epochs:
    :param learning_rate:
    :param X:
    :param Y:
    :return:
    """
    m, n = X.shape
    # 初始化权重
    W0 = np.random.random(m)
    Wt = W0
    k = 0
    loss_record = np.array([])
    for k in range(epochs):
        X, Y = random(X, Y)
        for batch in range(int(X.shape[1] / batch_size)):
            Xi = X[:, (batch * batch_size): (batch + 1) * batch_size]
            Yi = Y[(batch * batch_size): (batch + 1) * batch_size]
            loss = 0
            grad = np.zeros(Xi.shape[0])
            for i in range(Xi.shape[1]):
                loss += (np.dot(Wt, Xi[:, i]) - Yi[i]) ** 2
                grad += (np.dot(Wt, Xi[:, i]) - Yi[i]) * Xi[:, i]
            loss_record = np.append(loss_record, loss / Xi.shape[1])
            Wt = Wt - learning_rate * (2 * grad / Xi.shape[1])
            if np.linalg.norm(grad) < 1e-10:
                x = np.arange(k + 1)
                plt.plot(x, loss_record)
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.show()
                return Wt

    # x = np.arange(k + 1)
    # plt.plot(x, loss_record[])
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.show()
    return Wt


def plot(train_data, train_label, w, title):
    m, n = train_data.shape
    x1 = x2 = y1 = y2 = np.array([])
    for i in range(n):
        if train_label[i] == 1:
            x1 = np.append(x1, train_data[1][i])
            y1 = np.append(y1, train_data[2][i])
        elif train_label[i] == -1:
            x2 = np.append(x2, train_data[1][i])
            y2 = np.append(y2, train_data[2][i])
    P = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(x1, y1, s=30, c='red', marker='s', label='+1 data')
    ax.scatter(x2, y2, s=30, c='green', label='-1 data')
    a, b = np.amax(train_data[1, :]), np.amin(train_data[1, :])
    x = np.arange(b, a, 0.1)
    y = (-w[0] - w[1] * x) / w[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend(loc="upper left")
    plt.show()


def Accuracy(data, label, W):
    m, n = data.shape
    num = 0
    # 数据增广化
    # data = np.column_stack([1 * (np.ones(m)), data])
    for i in range(n):
        if work(data[:, i], W) == label[i]:
            num += 1
    return num / n


def work(x, w):
    a = sum(np.multiply(x, w))
    if a > 0:
        return 1
    elif a == 0:
        return 0
    else:
        return -1


def Train(mean1, mean2, cov, number):
    """

    :param mean1:
    :param mean2:
    :param cov:
    :param number:
    :return:
    """
    data1 = np.transpose(np.random.multivariate_normal(mean1, cov, number))
    data2 = np.transpose(np.random.multivariate_normal(mean2, cov, number))

    m, n = data1.shape
    # 数据增广化
    data1 = np.vstack([np.ones(n), data1])
    data2 = np.vstack([np.ones(n), data2])

    train_data = np.hstack([data1[:, 0:int(0.8 * number)], data2[:, 0:int(0.8 * number)]])
    train_label = np.hstack([np.ones(int(0.8 * number)), -1 * np.ones(int(0.8 * number))])
    test_data = np.hstack([data1[:, int(0.8 * number):], data2[:, int(0.8 * number):]])
    test_label = np.hstack([np.ones(int(0.2 * number)), -1 * np.ones(int(0.2 * number))])

    # 广义逆
    start_time1 = time.time()
    W = Gatter(train_data, train_label)
    end_time1 = time.time()
    plot(train_data, train_label, W, 'gatter_train')
    plot(test_data, test_label, W, 'gatter_test')

    print('使用广义逆算法的结果：W = (%.2f,%.2f,%.2f)' % (W[0], W[1], W[2]))
    acc_train_gatter = Accuracy(train_data, train_label, W)
    acc_test_gatter = Accuracy(test_data, test_label, W)
    print('在训练集上分类正确率：', acc_train_gatter, ', 在测试集上分类正确率：', acc_test_gatter)
    print('广义逆算法运行时间：%.6fms' % ((end_time1 - start_time1) * 1000))

    start_time2 = time.time()
    W2 = Grad(train_data, train_label, learning_rate=0.01, epochs=1000)
    # W2 = SGD(train_data, train_label, learning_rate=0.01, epochs=1000, batch_size=20)
    end_time2 = time.time()
    plot(train_data, train_label, W2, 'grad_train')
    plot(test_data, test_label, W2, 'grad_test')

    print('使用梯度下降法的结果：W = (%.2f,%.2f,%.2f)' % (W2[0], W2[1], W2[2]))
    acc_train_grad = Accuracy(train_data, train_label, W2)
    acc_test_grad = Accuracy(test_data, test_label, W2)
    print('在训练集上分类正确率：', acc_train_grad, ', 在测试集上分类正确率：', acc_test_grad)
    print('梯度下降算法运行时间：%.2f ms' % ((end_time2 - start_time2) * 1000))


cov = np.eye(2)
print('情况1：')
mean1 = np.array([-5, 0])
mean2 = np.array([0, 5])
Train(mean1, mean2, cov, number=200)

print('情况2：')
mean3 = np.array([1, 0])
mean4 = np.array([0, 1])
Train(mean3, mean4, cov, number=200)
