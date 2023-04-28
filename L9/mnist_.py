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
            optimizer.zero_grad()
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
