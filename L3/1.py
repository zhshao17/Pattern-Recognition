import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1, 0.2, 0.7],
              [1, 0.3, 0.3],
              [1, 0.4, 0.5],
              [1, 0.6, 0.5],
              [1, 0.1, 0.4],
              [1, 0.4, 0.6],
              [1, 0.6, 0.2],
              [1, 0.7, 0.4],
              [1, 0.8, 0.6],
              [1, 0.7, 0.5]])
# data1 = data.transpose()
# data2 = np.linalg.inv(data)

x1 = np.dot(np.linalg.inv((np.dot(x.transpose(), x))), x.transpose())
print(x1)
y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
w = np.dot(x1, y.transpose())
print(w)
y1 = []
for i in range(len(y)):
    y1.append(np.dot(x[i], w))
print(y1)
x1 = np.array([])
x2 = np.array([])
y1 = np.array([])
y2 = np.array([])
plt.figure(1)
for i in range(len(y)):
    if y[i] == 1:
        x1 = np.append(x1, x[i][1])
        y1 = np.append(y1, x[i][2])
    elif y[i] == -1:
        x2 = np.append(x2, x[i][1])
        y2 = np.append(y2, x[i][2])
plt.scatter(x1, y1, c='red', marker='s', label='+1 data')
plt.scatter(x2, y2, c='green', label='-1 data')
x_ = np.arange(0.4, 0.6, 0.01)
y = (-w[0] - w[1] * x_) / w[2]
plt.plot(x_, y)
plt.plot()
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc="upper left")
plt.title('W = (1.43,-3.22,0.24)')
plt.show()
