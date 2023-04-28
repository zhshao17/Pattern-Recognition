import time

import numpy as np
import matplotlib.pyplot as plt


def work(x, w):
    a = sum(np.multiply(x, w))
    if a > 0:
        return 1
    elif a == 0:
        return 0
    else:
        return -1


def plot(train_data, train_label, w, title):
    m, n = train_data.shape
    x1 = x2 = y1 = y2 = np.array([])
    for i in range(m):
        if train_label[i] == 1:
            x1 = np.append(x1, train_data[i][0])
            y1 = np.append(y1, train_data[i][1])
        elif train_label[i] == -1:
            x2 = np.append(x2, train_data[i][0])
            y2 = np.append(y2, train_data[i][1])
    P = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(x1, y1, s=30, c='red', marker='s', label='+1 data')
    ax.scatter(x2, y2, s=30, c='green', label='-1 data')
    a, b = np.amax(train_data[:, 0]), np.amin(train_data[:, 0])
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
    data = np.column_stack([1 * (np.ones(m)), data])
    for i in range(m):
        if work(data[i], W) == label[i]:
            num += 1
    return num / m


def PLA(train_data, train_label, w_0, x_0):
    """

    :param x_0:
    :param w_0:
    :param train_data: [[],[],[]]
    :param train_label: [,,]
    :return:
    """
    # 对权向量和train_data做增广化
    m, n = train_data.shape
    train_data = np.column_stack([x_0 * np.ones(m), train_data])  # m * (n+1)
    # 初始化权向量
    w0 = np.ones((n, 1), dtype=float)
    w0 = np.insert(w0, 0, w_0)
    # 进行循环
    k = 0
    w_record = w = w0
    while True:
        flag = True
        for i in range(m):
            if work(train_data[i], w) != train_label[i]:
                flag = False
                # 迭代w
                w = w + train_label[i] * train_data[i]
                w_record = np.vstack((w_record, w))
                break
        if flag or (k >= 1000):
            break
        k += 1
    return w_record


def Pocket(train_data, train_label, w_0, x_0, epochs):
    """

    :param train_data:
    :param train_label:
    :param w_0:
    :param x_0:
    :param epochs: 设定的迭代次数
    :return:
    """
    # 对权向量和train_data做增广化
    m, n = train_data.shape
    train_data = np.column_stack([x_0 * (np.ones(m)).T, train_data])  # m * (n+1)
    # 初始化权向量
    w0 = np.zeros((n, 1), dtype=float)
    w0 = np.insert(w0, 0, w_0)  # w0 = [w_0, w0]
    Wp = np.random.random(3)  # 随机给定义W
    w = w0
    Wp_record, w_record = Wp, w0
    for k in range(epochs):
        for i in range(m):
            if work(train_data[i], w) != train_label[i]:
                w = w + train_label[i] * train_data[i]
                # 比较
                Wp = GetBestW(train_data, train_label, w, Wp)
                Wp_record = np.vstack((Wp_record, Wp))
                w_record = np.vstack((w_record, w))
                break
                # Wp_record = np.append(Wp_record, Wp)
                # w_record = np.append(w_record, w)
    return Wp_record, w_record


def GetBestW(train_data, train_label, w, Wp):
    """
    比较w_{t+1} 和Wp错分的样本集数量
    :param train_data:
    :param train_label:
    :param w:
    :param Wp:
    :return:
    """
    m, n = train_data.shape
    num = numP = 0
    for i in range(m):
        if work(train_data[i], w) != train_label[i]:
            num += 1
        if work(train_data[i], Wp) != train_label[i]:
            numP += 1
    if num < numP:
        return w
    elif num >= numP:
        return Wp


def Train(mean1, mean2, cov, number):
    """

    :param mean1:
    :param mean2:
    :param cov:
    :param number:
    :return:
    """
    data1 = np.random.multivariate_normal(mean1, cov, number)
    data2 = np.random.multivariate_normal(mean2, cov, number)

    train_data = np.vstack([data1[:int(0.8 * number)], data2[:int(0.8 * number)]])
    train_label = np.hstack([np.ones(int(0.8 * number)), -1 * np.ones(int(0.8 * number))])
    test_data = np.vstack([data1[int(0.8 * number):], data2[int(0.8 * number):]])
    test_label = np.hstack([np.ones(int(0.2 * number)), -1 * np.ones(int(0.2 * number))])
    start_time1 = time.time()
    W_record = PLA(train_data, train_label, w_0=0, x_0=1)
    end_time1 = time.time()
    W = W_record[-1]
    plot(train_data, train_label, W, 'PLA_train')
    plot(test_data, test_label, W, 'PLA_test')

    print('使用PLA的结果：')
    print('第1000次迭代结束后, Wp is:', (W_record[-1][0], W_record[-1][1], W_record[-1][2]))
    acc_train_PLA = Accuracy(train_data, train_label, W)
    acc_test_PLA = Accuracy(test_data, test_label, W)
    print('在训练集上分类正确率：', acc_train_PLA, ', 在测试集上分类正确率：', acc_test_PLA)
    print('PLA算法运行时间：%.2fms' % ((end_time1 - start_time1) * 1000))

    start_time2 = time.time()
    Wp_record, w_record = Pocket(train_data, train_label, w_0=0, x_0=1, epochs=100)
    end_time2 = time.time()
    Wp = Wp_record[-1]
    plot(train_data, train_label, Wp, 'Pocket_train')
    plot(test_data, test_label, Wp, 'Pocket_test')

    print('使用Pocket的结果：')
    print('The best Wp is:', (Wp_record[-1][0], Wp_record[-1][1], Wp_record[-1][2]))
    acc_train_Pocket = Accuracy(train_data, train_label, Wp)
    acc_test_Pocket = Accuracy(test_data, test_label, Wp)
    print('在训练集上分类正确率：', acc_train_Pocket, ', 在测试集上分类正确率：', acc_test_Pocket)
    print('Pocket算法运行时间：%.2f ms' % ((end_time2 - start_time2) * 1000))


cov = np.eye(2)
print('情况1：')
mean1 = np.array([-5, 0])
mean2 = np.array([0, 5])
Train(mean1, mean2, cov, number=200)

print('情况2：')
mean3 = np.array([-1, 0])
mean4 = np.array([0, 1])
Train(mean3, mean4, cov, number=200)
