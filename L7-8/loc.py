import numpy as np
from SVM import PrimalSVM, DualSVM
import matplotlib.pyplot as plt
from func import acc, drawdiaoyu, drawSupport

"""
这里的样本数据从csdn上直接下载下来使用的，没有自己核实是否正确
"""
x1 = np.array([119.28, 26.08,  # 福州
               121.31, 25.03,  # 台北
               121.47, 31.23,  # 上海
               118.06, 24.27,  # 厦门
               121.46, 39.04,  # 大连
               122.10, 37.50,  # 威海
               124.23, 40.07]).reshape(7, 2)  # 丹东
x2 = np.array([129.87, 32.75,  # 长崎
               130.33, 31.36,  # 鹿儿岛
               131.42, 31.91,  # 宫崎
               130.24, 33.35,  # 福冈
               133.33, 15.43,  # 鸟取
               138.38, 34.98,  # 静冈
               140.47, 36.37]).reshape(7, 2)  # 水户

x3 = np.array([119.28, 26.08,  # 福州
               121.31, 25.03,  # 台北
               121.47, 31.23,  # 上海
               118.06, 24.27,  # 厦门
               121.46, 39.04,  # 大连
               122.10, 37.50,  # 威海
               124.23, 40.07,  # 丹东
               113.53, 29.58,  # 武汉-------
               104.06, 30.67,  # 成都-------
               116.25, 39.54,  # 北京-------
               ]).reshape(10, 2)
x4 = np.array([129.87, 32.75,  # 长崎
               130.33, 31.36,  # 鹿儿岛
               131.42, 31.91,  # 宫崎
               130.24, 33.35,  # 福冈
               133.33, 15.43,  # 鸟取
               138.38, 34.98,  # 静冈
               140.47, 36.37,  # 水户
               136.54, 35.10,  # 名古屋----
               132.27, 34.24,  # 广岛-----
               139.46, 35.42,  # 东京---
               ]).reshape(10, 2)

xtest = np.array([123.28, 25.44])
ytest = 1


def Train_Primal(x1, x2):
    m, n = x1.shape
    x = np.vstack([x1, x2])
    x = np.transpose(x)
    y = np.hstack([np.ones(m), -1 * np.ones(m)])
    b, w = PrimalSVM(x, y)
    w = np.array(w)
    w = w.reshape(n, )
    train_acc = acc(w, b, x, y)
    plt.figure()
    drawdiaoyu(x, y, w, b, 'train_Primal')
    print('b', b)
    print('w', w)
    print('train_acc', train_acc)
    if np.dot(w, xtest) + b > 0:
        print('钓鱼岛正确分类')
    plt.scatter(xtest[0], xtest[1], s=30, c='black', label='diaoyu')
    plt.legend(loc="upper left")
    plt.show()
    print('end')


def Train_Dual(x1, x2):
    m, n = x1.shape
    x = np.vstack([x1, x2])
    x = np.transpose(x)
    y = np.hstack([np.ones(m), -1 * np.ones(m)])
    b, w, alpha = DualSVM(x, y)
    train_acc = acc(w, b, x, y)
    plt.figure()
    drawdiaoyu(x, y, w, b, 'train')
    drawSupport(x, y, alpha)
    print('train_acc', train_acc)
    if np.dot(w, xtest) + b > 0:
        print('钓鱼岛正确分类')
    plt.scatter(xtest[0], xtest[1], s=30, c='black', label='diaoyu')
    plt.legend(loc="upper left")
    plt.show()


# Train_Primal(x1, x2)
# Train_Primal(x3, x4)
Train_Dual(x1, x2)
Train_Dual(x3, x4)