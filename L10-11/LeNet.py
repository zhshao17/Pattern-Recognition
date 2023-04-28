import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self, input_, hidden_, output_):
        """_summary_

        Args:
            input_ (_type_): 输入层的大小
            hidden_ (_type_): 隐藏层的大小，可以是一个[]
            output_ (_type_): 输出的类别数
        """
        super(Model, self).__init__()  # 自定义，注意参数与模型名称一致
        self.Conv1 = nn.Sequential(  # Sequential 用于将多个层组合在一起
            nn.Conv2d(in_channels=input_, out_channels=hidden_[0],
                      kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )
        self.AvePool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_[0], out_channels=hidden_[1],
                      kernel_size=5, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.AvePool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.FC1 = nn.Sequential(
            nn.Flatten(),  # 将图像展成一维
            nn.Linear(400, 120),
            nn.Sigmoid()
        )
        self.FC2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.Sigmoid()
        )
        self.Linear = nn.Linear(84, output_)
        self.Softmax = nn.Softmax()

    def forword(self, x):
        x = self.Conv1(x)
        x = self.AvePool1(x)
        x = self.Conv2(x)
        x = self.AvePool2(x)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.Linear(x)
        x = self.Softmax(x)
        return x


def GetMnist(batch):
    """读取数据，读取的路径就在本文件路径下，如果没有，“download=True”表示直接从网上下载

    Args:
        batch (int): 批次大小

    Returns:
        _type_: _description_
    """
    train_dataset = datasets.MNIST(root='./', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    data_loader_train = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
    data_loader_test = DataLoader(dataset=test_dataset, batch_size=batch, shuffle=False)
    print('Get data!')
    return data_loader_train, data_loader_test


net = Model(input_=1, hidden_=[6, 16], output_=10)  # 构建一个自己的网络，并设定模型结构参数（不训练的参数）
loss = torch.nn.CrossEntropyLoss()  # 定义损失函数，这里因为是分类问题所以用的交叉熵
optimizer = torch.optim.Adam(Model.parameters(self=net), lr=0.01)  # 定义优化器，如SGD，Adam等


def train(Net, epochs, batch_size, loss, op):
    """_summary_

    Args:
        Net (_type_): 网络
        epochs (_type_):训练次数，每一个epoch遍历所有数据
        batch_size (_type_): 批次
        loss (_type_): 损失函数
        op (_type_): 优化器
    """
    traindata, testdata = GetMnist(batch_size)  # 读取数据
    train_acc_re = np.array([])  # 记录训练集精确度
    train_loss_re = np.array([])  # 记录训练集损失
    test_acc_re = np.array([])  # 记录测试集精确度
    for epoch in range(epochs + 1):
        train_acc = 0
        train_loss = 0
        n = 0
        Net.train()
        for X, Y in traindata:
            out = Net.forword(X)  # 前项计算网络输出
            l = loss(out, Y).sum()  # 计算损失值
            op.zero_grad()  # 梯度置为0
            l.backward()  # 反向计算
            op.step()  # 优化模型参数

            train_acc += (out.argmax(dim=1) == Y).sum().item()
            train_loss += l.item()
            n += Y.shape[0]
        print('******epoch %d:   loss %.5f, train acc %.3f %%' % (epoch + 1, train_loss / n, (train_acc / n) * 100))

        train_acc_re = np.append(train_acc_re, (train_acc / n) * 100)
        train_loss_re = np.append(train_loss_re, train_loss / n)

        test_acc_count = 0
        n = 0
        Net.eval()
        for X, Y in testdata:
            out = Net.forword(X)
            test_acc_count += (out.argmax(dim=1) == Y).sum().item()
            n += Y.shape[0]
        print('*************test acc %.3f %%' % (test_acc_count * 100 / n))
        test_acc_re = np.append(test_acc_re, test_acc_count * 100 / n)

    x = np.array(range(epochs + 1))
    plt.figure(1)
    plt.plot(x, train_acc_re)
    plt.xlabel('epoch')
    plt.ylabel('train_acc/%')
    plt.grid()
    plt.show()

    plt.figure(2)
    plt.plot(x, train_loss_re * 100)
    plt.xlabel('epoch')
    plt.ylabel('train_loss/10^{-2}')
    plt.grid()
    plt.show()

    plt.figure(3)
    plt.plot(x, test_acc_re)
    plt.xlabel('epoch')
    plt.ylabel('test_acc/%')
    plt.grid()
    plt.show()

    # # 随机抽取10个测试集样本
    # test_dataset = datasets.MNIST(root='./', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    # data_test = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)
    #
    # for X, Y in data_test:
    #     out = net.forword(X)
    #     k = out.argmax(dim=1)
    #     print('分类结果：')
    #     print(k)
    #     print('实际结果')
    #     print(Y)
    #     print((out.argmax(dim=1) == Y))
    #     break


train(Net=net, epochs=10, batch_size=256, loss=loss, op=optimizer)
