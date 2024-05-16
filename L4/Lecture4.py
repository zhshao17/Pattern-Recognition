import numpy as np


X1 = np.array([[5, 7, 10, 11.5, 14, 12],
               [37, 30, 35, 40, 38, 31]])
X2 = np.array([[35, 39, 34, 37],
               [21.5, 21.7, 16, 17]])
miu1 = np.array([np.sum(X1[0, :])/6, np.sum(X1[1, :])/6])
miu2 = np.array([np.sum(X2[0, :])/4, np.sum(X2[1, :])/4])
print(miu1)
print(miu2)
sigma1 = np.array([[0, 0],
                  [0, 0]])
sigma2 = np.array([[0, 0],
                  [0, 0]])
for i in range(6):
    b = np.outer(X1[:, i] - miu1.T, np.transpose(X1[:, i] - miu1.T))
    sigma1 = sigma1 + b
for i in range(4):
    b = np.outer(X2[:, i] - miu2.T, np.transpose(X2[:, i] - miu2.T))
    sigma2 = sigma2 + b
print(sigma1)
print(sigma2)
S_w = sigma1 + sigma2
print(S_w)
S_w1 = np.linalg.inv(S_w)
print(S_w1)
W = np.dot(S_w1, (np.transpose(miu1 - miu2)))
print(W)
s = np.dot(W, (miu1 + miu2))/2
print(s)