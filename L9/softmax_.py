def softmax(X, Y):
    w1 = np.ones(4)
    w2 = np.ones(4)
    w3 = np.ones(4)
    k = 0
    while True:
        # np.random.shuffle(traindata, )
        flag = True
        for i in range(len(Y)):
            y1 = math.exp(np.dot(w1, X[:, i]))
            y2 = math.exp(np.dot(w2, X[:, i]))
            y3 = math.exp(np.dot(w3, X[:, i]))
            y = y1 + y2 + y3
            y1 = y1 / y
            y2 = y2 / y
            y3 = y3 / y
            j = np.argmax(np.array([y1, y2, y3])) + 1
            if j != Y[i]:
                w1 = update(w1, y1, Y[i], 1, X[:, i], lr=0.01)
                w2 = update(w2, y2, Y[i], 2, X[:, i], lr=0.01)
                w3 = update(w3, y3, Y[i], 3, X[:, i], lr=0.01)
                flag = False
                break
        if flag or (k >= 10000):
            break
        k += 1
    return w1, w2, w3


def acc(w1, w2, w3, X, Y):
    Y1 = np.array([])
    num = 0
    for i in range(len(Y)):
        y = getY(w1, w2, w3, X[:, i])
        Y1 = np.append(Y1, y)
        if y == Y[i]:
            num += 1
    return num / len(Y), Y1

