def Fisher(X1, X2):
    """
    X1, X2 均按列向量
    :param X1:
    :param X2:
    :return:
    """
    m1, n1 = X1.shape
    m2, n2 = X2.shape
    miu1 = np.array([np.sum(X1[0, :]) / n1, np.sum(X1[1, :]) / n1])
    miu2 = np.array([np.sum(X2[0, :]) / n2, np.sum(X2[1, :]) / n2])
    sigma1 = np.zeros([m1, m1])
    sigma2 = np.zeros([m2, m2])
    for i in range(n1):
        b = np.outer(X1[:, i] - miu1.T, np.transpose(X1[:, i] - miu1.T))
        sigma1 = sigma1 + b
    for i in range(n2):
        b = np.outer(X2[:, i] - miu2.T, np.transpose(X2[:, i] - miu2.T))
        sigma2 = sigma2 + b
    S_w = sigma1 + sigma2
    S_w1 = np.linalg.inv(S_w)
    W = np.dot(S_w1, (np.transpose(miu1 - miu2)))
    s = np.dot(W, (miu1 + miu2)) / 2
    return W, s