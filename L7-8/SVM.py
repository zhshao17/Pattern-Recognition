import numpy as np
from cvxopt import solvers, matrix
import math


def PrimalSVM(X, Y):
    """

    :param X: x_n是列向量
    :param Y:
    :return:
    """
    d, m = X.shape
    P = np.eye(d)  # PPT上的Q
    P = np.vstack([np.zeros(d), P])
    P = np.hstack([np.zeros([d + 1, 1]), P])
    q = np.zeros(d + 1)  # PPT上的q
    G = np.empty((0, 3), float)  # PPT上的A
    for i in range(m):
        a = X[:, i]
        a = np.insert(a, 0, 1)
        am_T = -1 * Y[i] * a
        G = np.vstack([G, am_T])
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(-1 * np.ones(m))  # PPT 上的c
    sol = solvers.qp(P, q, G, h, A=None, b=None)
    u = sol['x']
    b = u[0]
    w = u[1:]
    return b, w


def DualSVM(X, Y):
    d, m = X.shape
    N = m
    P = np.empty([N, N])  # PPT上的Q
    for i in range(N):
        for j in range(N):
            P[i, j] = Y[i] * Y[j] * np.dot(X[:, i], X[:, j])
    P = matrix(P)
    q = matrix(-1 * np.ones(N))  # PPT上的p
    G = matrix(-1 * np.eye(N))  # PPT上的A
    h = matrix(np.zeros([N, 1]))  # PPT上的c
    A = matrix(Y)
    A = A.T  # PPT上的r
    b = matrix(np.zeros(1))  # PPT上的v
    """
            minimize    (1/2)*x'*P*x + q'*x
            subject to  G*x <= h
                        A*x = b.
    """
    sol = solvers.qp(P, q, G, h, A, b)
    alpha = np.array(sol['x'])
    w = np.zeros(2)
    for i in range(N):
        if alpha[i] >= 0:
            w += alpha[i] * Y[i] * X[:, i]
    # 选择一个大于0的alpha，这里直接选择最大的那个，且只需要判断是否大于0就行
    if alpha[np.argmax(alpha)] > 1e-6:
        b = np.dot(w, X[:, np.argmax(alpha)])
        b_ = Y[np.argmax(alpha)] - b
        return b_, w, alpha
    else:
        print('alpha < 0!!!!!!')
        return 0, 0, 0


def kernelSVM(X, Y, kenel):
    """

    :param X:
    :param Y:
    :return:
    """
    d, m = X.shape
    N = m
    P = np.empty([N, N], dtype=np.double)  # PPT上的Q
    for i in range(N):
        for j in range(N):
            if kenel == 'four':
                P[i, j] = Y[i] * Y[j] * math.pow(np.dot(X[:, i].T, X[:, j]) + 1, 4)
            elif kenel == 'gao':
                # 高斯
                P[i, j] = Y[i] * Y[j] * np.exp(-1 *np.linalg.norm(X[:, i] - X[:, j]) ** 2)
    P = matrix(P)
    q = matrix(-1 * np.ones(N))  # PPT上的p
    G = matrix(-1 * np.eye(N))  # PPT上的A
    h = matrix(np.zeros([N, 1]))  # PPT上的c
    A = matrix(Y)
    A = A.T  # PPT上的r
    b = matrix(np.zeros(1))  # PPT上的v

    sol = solvers.qp(P, q, G, h, A, b)
    alpha = np.array(sol['x']).reshape([480, ])
    return alpha