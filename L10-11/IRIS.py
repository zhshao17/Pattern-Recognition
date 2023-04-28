import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def GetData():
    Data = pd.read_csv('iris.csv').values
    index = np.arange(150)

    index1 = random.sample(range(50), 30)
    index2 = random.sample(range(50), 30) + np.array(50)
    index3 = random.sample(range(50), 30) + np.array(100)

    index_train = np.hstack([index1, index2, index3])
    index_test = np.delete(index, index_train)

    traindata = Data[index_train, :]
    testdata = Data[index_test, :]
    train_dataset = []
    test_dataset = []
    for i in range(traindata.shape[0]):
        x = traindata[i, 0:4]
        y = traindata[i, 4:7]
        train_dataset.append([x, y])
    for i in range(testdata.shape[0]):
        x = testdata[i, 0:4]
        y = testdata[i, 4:7]
        test_dataset.append([x, y])
    train_dataset = MyDataset(train_dataset)
    test_dataset = MyDataset(test_dataset)
    train_data = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=10, shuffle=True)
    return train_data, test_data


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.cov1 = nn.Sequential(
            nn.Linear(in_features=4, out_features=8),
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Tanh(),
            nn.Dropout()
        )
        self.cov2 = nn.Sequential(
            nn.Linear(in_features=8, out_features=10),
            nn.ReLU(),
            # nn.Tanh()
            # nn.Sigmoid()
            # nn.Dropout()
        )
        # self.cov3 = nn.Sequential(
        #     nn.Linear(in_features=4, out_features=3),
        #     nn.ReLU(),
        # )
        self.Linear = nn.Linear(in_features=10, out_features=3)

    def forward(self, x):
        x = self.cov1(x)
        x = self.cov2(x)
        # x = self.cov3(x)
        x = self.Linear(x)
        x = F.softmax(x, dim=1)
        return x


def accuracy(x, t):
    x = np.array(x)
    l = x.size
    # x = np.argmax(x, axis=1)
    if t.ndim != 1:
        t = np.argmax(t, axis=1)
    Accuracy = np.sum(x == t) / l  # float(x.shape[0])
    return Accuracy


epochs = 100
learning_rate = 0.01

model = model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_data, test_data = GetData()
Acc = []
List = []
Acc_test = []
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    loss_sum = 0
    i = 0
    acc_sum = 0
    for data_x, data_y in train_data:
        i += 1
        optimizer.zero_grad()
        data_x = data_x.to(torch.float32)
        data_y = data_y.to(torch.float32)
        outputs = model(data_x)
        loss = criterion(data_y, outputs)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        acc = accuracy(torch.max(outputs, 1)[1], np.array(data_y))
        acc_sum += acc

    print('epoch:', epoch, 'loss', loss_sum / i, 'acc', acc_sum / i)
    List.append(loss_sum / i)
    Acc.append(acc_sum / i)

    # =====================valid============================
    model.eval()
    test_acc = 0
    i = 0
    for data_x, data_y in test_data:
        i += 1
        data_x = data_x.to(torch.float32)
        data_y = data_y.to(torch.float32)
        outputs = model(data_x)
        loss = criterion(outputs, data_y)

        acc = accuracy(torch.max(outputs, 1)[1], np.array(data_y))
        test_acc += acc
    Acc_test.append(test_acc / i)

plt.figure(1)
plt.plot(range(epochs), List)
plt.xlabel('epoch')
plt.ylabel('train_loss')
plt.show()
plt.figure(2)
plt.plot(range(epochs), Acc)
plt.xlabel('epoch')
plt.ylabel('train_Acc')
plt.show()
plt.figure(3)
plt.plot(range(epochs), Acc_test)
plt.xlabel('epoch')
plt.ylabel('test_acc')
plt.show()
