import math
import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def load_mnist_train(path, kind='train'):
    """

    :param path: 数据集的路径
    :param kind: 为train，代表读取训练集
    :return:
    """
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    # 不再用gzip打开文件
    with open(labels_path, 'rb') as lbpath:
        # 使用struct.unpack方法读取前两个数据，>代表高位在前，I代表32位整型。lbpath.read(8)表示一次从文件中读取8个字节, 这样读到的前两个数据分别是magic number和样本个数
        magic, n = struct.unpack('>II', lbpath.read(8))
        # 使用np.fromfile读取剩下的数据
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def batch_generator(all_data, batch_size, shuffle=True):
    """
    :param all_data : all_data整个数据集，包含输入和输出标签
    :param batch_size: batch_size表示每个batch的大小
    :param shuffle: 是否打乱顺序
    :return:
    """
    # 输入all_datas的每一项必须是numpy数组，保证后面能按p所示取值
    all_data = [np.array(d) for d in all_data]
    # 获取样本大小
    data_size = all_data[0].shape[0]
    # print("data_size: ", data_size)
    if shuffle:
        # 随机生成打乱的索引
        p = np.random.permutation(data_size)
        # 重新组织数据
        all_data = [d[p] for d in all_data]
    batch_count = 0
    while True:
        # 数据一轮循环(epoch)完成，打乱一次顺序
        if batch_count * batch_size + batch_size > data_size:
            batch_count = 0
            if shuffle:
                p = np.random.permutation(data_size)
                all_data = [d[p] for d in all_data]
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start: end] for d in all_data]


epochs = 10
batch_size = 256
learning_rate = 0.00001
train_image, train_label = load_mnist_train(path='./MNIST', kind='train')
test_image, test_label = load_mnist_train(path='./MNIST', kind='t10k')

batch_num = int(len(train_label) / batch_size)
W = np.random.normal(loc=0.0, scale=0.01, size=784 * 10).reshape(784, 10)  # 784 * 10
# W = np.zeros([784, 10])
# 计算测试集精确度
S_test = np.dot(test_image, W)  # 10000 * 10
expS_test = np.exp(S_test)
Yhat_test = expS_test / expS_test.sum(axis=1, keepdims=True)  # 10000 * 10
Y_test = np.argmax(Yhat_test,  axis=1)
num = 0
for i in range(len(Y_test)):
    if train_label[i] == Y_test[i]:
        num += 1
print('初始分类精确度：', num/len(Y_test) * 100, '%')

Lin_record = np.array([])

for epoch in range(epochs):
    batch_gen = batch_generator([train_image, train_label], batch_size=batch_size, shuffle=True)
    loss_ = 0
    for num in range(batch_num):
        batch_x, batch_y = next(batch_gen)
        # 生成Y
        Y = np.zeros([batch_size, 10])
        for j in range(batch_size):
            Y[j, batch_y[j]] = 1
        # print(batch_x.shape)  # 256*784
        S = np.dot(batch_x, W)  # 256 * 10
        # print(S.shape)
        expS = np.exp(S)
        Yhat = expS / expS.sum(axis=1, keepdims=True)  # 256 * 10
        # 计算Lin
        Lin = 0
        for k in range(batch_size):
            Lin += -1 * np.log(Yhat[k, batch_y[k]])
        Lin /= batch_size
        loss_ += Lin
        # print('epoch', epoch, 'num', num+1, 'loss =', Lin)
        Lin_record = np.append(Lin_record, Lin)
        # 求梯度
        Yhat_Y = np.transpose(Yhat - Y)
        delW = np.dot(Yhat_Y, batch_x)
        # 更新
        W = W - learning_rate * np.transpose(delW)
    print('------------------epoch', epoch, 'loss', loss_/batch_num)
S_test = np.dot(test_image, W)  # 10000 * 10
expS_test = np.exp(S_test)
Yhat_test = expS_test / expS_test.sum(axis=1, keepdims=True)  # 10000 * 10
Y_test = np.argmax(Yhat_test,  axis=1)
num = 0
for i in range(len(Y_test)):
    if train_label[i] == Y_test[i]:
        num += 1
print('结束分类精确度：', num/len(Y_test) * 100, '%')
# plt.figure()
# plt.plot(range(len(Lin_record)), Lin_record)
# plt.show()
