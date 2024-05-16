def Pocket(train_data, train_label, w_0, x_0, epochs):
    """

    :param train_data:
    :param train_label:
    :param w_0:
    :param x_0:
    :param epochs: 设定的迭代次数
    :return:
    """
    m, n = train_data.shape
    train_data = np.column_stack([x_0 * (np.ones(m)).T, train_data])  # m * (n+1)，增广化
    # 初始化权向量
    w0 = np.zeros((n, 1), dtype=float)
    w0 = np.insert(w0, 0, w_0)  # w0 = [w_0, w0],增广化
    Wp = np.random.random(3)  # 随机给定义W
    w = w0
    Wp_record, w_record = Wp, w0
    for k in range(epochs):
        for i in range(m):
            if work(train_data[i], w) != train_label[i]:
                w = w + train_label[i] * train_data[i]
                # 比较
                Wp = GetBestW(train_data, train_label, w, Wp)
                Wp_record = np.vstack((Wp_record, Wp))
                w_record = np.vstack((w_record, w))
                break
    return Wp_record, w_record


def GetBestW(train_data, train_label, w, Wp):
    """
    比较w_{t+1} 和Wp错分的样本集数量
    :param train_data:
    :param train_label:
    :param w:
    :param Wp:
    :return:
    """
    m, n = train_data.shape
    num = numP = 0
    for i in range(m):
        if work(train_data[i], w) != train_label[i]:
            num += 1
        if work(train_data[i], Wp) != train_label[i]:
            numP += 1
    if num < numP:
        return w
    else:
        return Wp


def work(x, w):
    a = sum(np.multiply(x, w))
    if a > 0:
        return 1
    elif a == 0:
        return 0
    else:
        return -1
