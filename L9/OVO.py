import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import random


def CreateData():
    Data = pd.read_csv('./Iris数据集/iris.csv', header=0, usecols=[1, 2, 3, 4, 5])
    Data1 = np.array(Data[0:50])
    Data2 = np.array(Data[50:100])
    Data3 = np.array(Data[100:])
    index = np.arange(50)
    index11 = random.sample(range(50), 30)
    index21 = np.delete(index, index11)
    index12 = random.sample(range(50), 30)
    index22 = np.delete(index, index12)
    index13 = random.sample(range(50), 30)
    index23 = np.delete(index, index13)
    traindata = np.vstack([Data1[index11, :], Data2[index12, :], Data3[index13, :]])
    testdata = np.vstack([Data1[index21, :], Data2[index22, :], Data3[index23, :]])
    return traindata, testdata


def work(x, w):
    a = np.dot(x, w)
    if a > 0:
        return 1
    elif a == 0:
        return 0
    else:
        return -1


def PLA(train_data, train_label, w_0, x_0):
    """

    :param x_0:
    :param w_0:
    :param train_data: 这里用的行向量
    :param train_label: [,,]
    :return:
    """
    # 对权向量和train_data做增广化
    m, n = train_data.shape
    train_data = np.column_stack([x_0 * np.ones(m), train_data])  # m * (n+1)
    # 初始化权向量
    w0 = np.zeros((n, 1), dtype=float)
    w0 = np.insert(w0, 0, w_0)
    # 进行循环
    k = 0
    w = w0
    while True:
        flag = True
        for i in range(m):
            if work(train_data[i], w) != train_label[i]:
                flag = False
                # 迭代w
                w = w + train_label[i] * train_data[i]
                break
        if flag or (k >= 10000):
            break
        k += 1
    return w


def OVO(Data):
    # X1 = np.transpose(Data[0:50, :-2])
    # X2 = np.transpose(Data[50:50, :-2])
    # X3 = np.transpose(Data[0:50, :-2])
    X1 = Data[0:30, :-1]
    X2 = Data[30:60, :-1]
    X3 = Data[60:90, :-1]

    # Y1 = Data[0:50, -1]
    # Y2 = Data[50:100, -1]
    # Y3 = Data[100:, -1]

    X12 = np.vstack([X1, X2])
    Y12 = np.hstack([np.ones(30), -1 * np.ones(30)])
    Data12 = np.hstack([X12, Y12.reshape(60, 1)])
    np.random.shuffle(Data12)
    X12 = Data12[:, 0:4]
    Y12 = Data12[:, -1]
    w12 = PLA(X12, Y12, w_0=0, x_0=1)
    #
    X13 = np.vstack([X1, X3])
    Y13 = np.hstack([np.ones(30), -1 * np.ones(30)])
    Data13 = np.hstack([X13, Y13.reshape(60, 1)])
    np.random.shuffle(Data13)
    X13 = Data13[:, 0:4]
    Y13 = Data13[:, -1]
    w13 = PLA(X13, Y13, w_0=0, x_0=1)
    #
    X23 = np.vstack([X2, X3])
    Y23 = np.hstack([np.ones(30), -1 * np.ones(30)])
    Data23 = np.hstack([X23, Y23.reshape(60, 1)])
    np.random.shuffle(Data23)
    X23 = Data23[:, 0:4]
    Y23 = Data23[:, -1]
    w23 = PLA(X23, Y23, w_0=0, x_0=1)
    return w12, w13, w23


def acc(X, Y, w12, w13, w23):
    """

    :param X:
    :param Y:
    :param w12:
    :param w13:
    :param w23:
    :return:
    """
    m, n = X.shape
    num = 0
    # 数据增广化
    X = np.column_stack([1 * (np.ones(m)), X])
    Y1 = np.array([])
    num = 0
    for i in range(m):
        if work(X[i, :], w12) > 0:
            y12 = 1
        else:
            y12 = 2
        if work(X[i, :], w13) > 0:
            y13 = 1
        else:
            y13 = 3
        if work(X[i, :], w23) > 0:
            y23 = 2
        else:
            y23 = 3
        # 投票
        y = np.argmax(np.bincount(np.array([y12, y13, y23])))
        Y1 = np.append(Y1, y)
        if y == Y[i]:
            num += 1
    return Y1, num / len(Y)


def OVOTrain():
    traindata, testdata = CreateData()
    trainlabel = np.hstack([np.ones(30), np.ones(30) * 2, np.ones(30) * 3])
    testlabel = np.hstack([np.ones(20), np.ones(20) * 2, np.ones(20) * 3])
    w12, w13, w23 = OVO(traindata)
    Ytrain, train_acc = acc(traindata[:, 0:4], trainlabel, w12, w13, w23)
    Ytest, test_acc = acc(testdata[:, 0:4], testlabel, w12, w13, w23)
    print('得到的类1与类2的分类面：', w12)
    print('得到的类1与类3的分类面：', w13)
    print('得到的类2与类3的分类面：', w23)
    print('训练集精度{:2f}%,测试集精度{:2f}%'.format(train_acc * 100, test_acc * 100))
    print(Ytrain)
    print(trainlabel)
    print(Ytest)
    print(testlabel)


OVOTrain()
