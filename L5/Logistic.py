def Logistic(X, Y, batch_size, epochs, learning_rate):
    """

    :param learning_rate:
    :param X:
    :param Y:
    :param batch_size:
    :param epochs:
    :return:
    """
    m, n = X.shape
    Wt = np.random.random(m)
    loss_record = np.array([])
    for epoch in range(epochs):
        X, Y = random(X, Y)
        for i in range(int(n / batch_size)):
            Xi = X[:, i * batch_size:(i + 1) * batch_size]
            Yi = Y[i * batch_size:(i + 1) * batch_size]
            # 计算一个batch的grad
            grad = np.zeros(m)
            for k in range(batch_size):
                grad += sigmoid(-1 * Yi[k] * np.dot(Wt, Xi[:, k])) * (-1 * Yi[k] * Xi[:, k])
            # 更新权向量W
            Wt = Wt - learning_rate * grad / batch_size
            if np.sum(grad) < 1e-10:
                return Wt
        # 计算总样本loss
        loss = 0
        for j in range(len(Y)):
            loss += math.log((1 + math.exp(-1 * Y[j] * np.dot(Wt, X[:, j]))), math.e)
        loss_record = np.append(loss_record, loss / len(Y))
    plt.figure()
    plt.plot(np.arange(epochs), loss_record)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    return Wt