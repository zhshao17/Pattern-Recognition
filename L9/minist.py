import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision
import matplotlib.pyplot as plt

train_dataset = datasets.MNIST(root='./', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./', train=False, transform=torchvision.transforms.ToTensor(), download=True)

data_loader_train = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
data_loader_test = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

print(data_loader_train)
print(data_loader_test)


class SoftmaxMnist(torch.nn.Module):
    def __init__(self, inputs, outputs):
        super(SoftmaxMnist, self).__init__()
        self.linear = torch.nn.Linear(inputs, outputs)

    def forword(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y


net = SoftmaxMnist(inputs=784, outputs=10)

# 权向量初始值由均值为0、标准差为0.01的正态分布产生的随机数得到
torch.nn.init.normal_(net.linear.weight, mean=0.0, std=0.01)
torch.nn.init.constant_(net.linear.bias, val=0)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.linear.parameters(), lr=0.01)


def MnistSoftmax(epochs, traindata, testdata, Net, loss, op, ):
    train_acc_re = np.array([])
    train_loss_re = np.array([])
    test_acc_re = np.array([])
    for epoch in range(epochs + 1):
        train_acc = 0
        train_loss = 0
        n = 0
        for X, Y in traindata:
            out = Net.forword(X)
            l = loss(out, Y).sum()
            op.zero_grad()
            l.backward()
            op.step()

            train_acc += (out.argmax(dim=1) == Y).sum().item()
            train_loss += l.item()
            n += Y.shape[0]
        print('******epoch %d:   loss %.5f, train acc %.3f %%' % (epoch + 1, train_loss / n, (train_acc / n) * 100))

        train_acc_re = np.append(train_acc_re, (train_acc / n) * 100)
        train_loss_re = np.append(train_loss_re, train_loss / n)

        test_acc_count = 0
        n = 0
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
    plt.ylabel('train_acc')
    plt.grid()
    plt.show()

    plt.figure(2)
    plt.plot(x, train_loss_re)
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.grid()
    plt.show()

    plt.figure(3)
    plt.plot(x, test_acc_re)
    plt.xlabel('epoch')
    plt.ylabel('test_acc')
    plt.grid()
    plt.show()

    # 随机抽取10个测试集样本

    data_test = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)

    for X, Y in data_test:
        out = net.forword(X)
        k = out.argmax(dim=1)
        print('分类结果：')
        print(k)
        print('实际结果')
        print(Y)
        print((out.argmax(dim=1) == Y))
        break


MnistSoftmax(epochs=10, traindata=data_loader_train, testdata=data_loader_test, Net=net, loss=loss, op=optimizer)

