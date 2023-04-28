def OVO(Data):
    X1 = Data[0:30, :-1]
    X2 = Data[30:60, :-1]
    X3 = Data[60:90, :-1]

    X12 = np.vstack([X1, X2])
    Y12 = np.hstack([np.ones(30), -1 * np.ones(30)])
    Data12 = np.hstack([X12, Y12.reshape(60, 1)])
    np.random.shuffle(Data12)
    X12 = Data12[:, 0:4]
    Y12 = Data12[:, -1]
    w12 = PLA(X12, Y12, w_0=0, x_0=1)
    #
    X13 = np.vstack([X1, X3])
    Y13 = np.hstack([np.ones(30), -1 * np.ones(30)])
    Data13 = np.hstack([X13, Y13.reshape(60, 1)])
    np.random.shuffle(Data13)
    X13 = Data13[:, 0:4]
    Y13 = Data13[:, -1]
    w13 = PLA(X13, Y13, w_0=0, x_0=1)
    #
    X23 = np.vstack([X2, X3])
    Y23 = np.hstack([np.ones(30), -1 * np.ones(30)])
    Data23 = np.hstack([X23, Y23.reshape(60, 1)])
    np.random.shuffle(Data23)
    X23 = Data23[:, 0:4]
    Y23 = Data23[:, -1]
    w23 = PLA(X23, Y23, w_0=0, x_0=1)
    return w12, w13, w23


def acc(X, Y, w12, w13, w23):
    m, n = X.shape
    num = 0
    # 数据增广化
    X = np.column_stack([1 * (np.ones(m)), X])
    Y1 = np.array([])
    num = 0
    for i in range(m):
        if work(X[i, :], w12) > 0:
            y12 = 1
        else:
            y12 = 2
        if work(X[i, :], w13) > 0:
            y13 = 1
        else:
            y13 = 3
        if work(X[i, :], w23) > 0:
            y23 = 2
        else:
            y23 = 3
        # 投票
        y = np.argmax(np.bincount(np.array([y12, y13, y23])))
        Y1 = np.append(Y1, y)
        if y == Y[i]:
            num += 1
    return Y1, num / len(Y)