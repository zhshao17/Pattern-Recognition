import math

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
    testdata = np.vstack([Data1[index21, :], Data2[index23, :], Data3[index23, :]])
    return traindata, testdata


def update(w, y, k, j, x, lr):
    if j == k:
        w = w - (y - 1) * x * lr
    else:
        w = w - y * x * lr
    return w


def getY(w1, w2, w3, X):
    y1 = math.exp(np.dot(w1, X))
    y2 = math.exp(np.dot(w2, X))
    y3 = math.exp(np.dot(w3, X))
    y = y1 + y2 + y3
    y1 = y1 / y
    y2 = y2 / y
    y3 = y3 / y
    return np.argmax(np.array([y1, y2, y3])) + 1


def acc(w1, w2, w3, X, Y):
    Y1 = np.array([])
    num = 0
    for i in range(len(Y)):
        y = getY(w1, w2, w3, X[:, i])
        Y1 = np.append(Y1, y)
        if y == Y[i]:
            num += 1
    return num / len(Y), Y1


def softmax(X, Y):
    w1 = np.ones(4)
    w2 = np.ones(4)
    w3 = np.ones(4)
    k = 0
    while True:
        # np.random.shuffle(traindata, )
        flag = True
        for i in range(len(Y)):
            y1 = math.exp(np.dot(w1, X[:, i]))
            y2 = math.exp(np.dot(w2, X[:, i]))
            y3 = math.exp(np.dot(w3, X[:, i]))
            y = y1 + y2 + y3
            y1 = y1 / y
            y2 = y2 / y
            y3 = y3 / y
            j = np.argmax(np.array([y1, y2, y3])) + 1
            if j != Y[i]:
                w1 = update(w1, y1, Y[i], 1, X[:, i], lr=0.01)
                w2 = update(w2, y2, Y[i], 2, X[:, i], lr=0.01)
                w3 = update(w3, y3, Y[i], 3, X[:, i], lr=0.01)
                flag = False
                break
        if flag or (k >= 10000):
            break
        k += 1
    return w1, w2, w3


def softmaxTrain():
    traindata, testdata = CreateData()

    traindata = np.transpose(traindata)
    trainX = traindata[0:4, :]
    trainY = np.hstack([np.ones(30), np.ones(30) * 2, np.ones(30) * 3])

    testdata = np.transpose(testdata)
    testX = testdata[0:4, :]
    testY = np.hstack([np.ones(20), np.ones(20) * 2, np.ones(20) * 3])

    w1, w2, w3 = softmax(trainX, trainY)
    train_acc, Ytrain = acc(w1, w2, w3, trainX, trainY)
    test_acc, Ytest = acc(w1, w2, w3, testX, testY)
    print('训练集精度{:2f}%,测试集精度{:2f}%'.format(train_acc * 100, test_acc * 100))
    print(trainY)
    print('训练集分类结果：\n', Ytrain)
    print(testY)
    print('测试集分类结果：\n', Ytest)


softmaxTrain()
