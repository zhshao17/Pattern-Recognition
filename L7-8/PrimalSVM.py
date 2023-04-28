import numpy as np
from cvxopt import solvers, matrix
import matplotlib.pyplot as plt
from func import draw, acc, CreatData
from SVM import PrimalSVM


def PrimalTrain(mean1, mean2):
    cov = np.eye(2)
    traindata, trainlabel, testdata, testlabel = CreatData(mean1, mean2, cov, number=200)
    b, w = PrimalSVM(traindata, trainlabel)

    train_acc = acc(w, b, traindata, trainlabel)
    test_acc = acc(w, b, testdata, testlabel)
    plt.figure()
    draw(traindata, trainlabel, w, b, 'train')
    plt.legend(loc="upper left")
    # plt.savefig('./image/Primal_train.png')
    plt.show()
    plt.figure()
    draw(testdata, testlabel, w, b, 'test')
    plt.legend(loc="upper left")
    # plt.savefig('./image/Primal_test.png')
    plt.show()
    print('b', b)
    print('w', w)
    print('train_acc', train_acc, 'test_acc', test_acc)


mean1 = np.array([-5, 0])
mean2 = np.array([0, 5])
PrimalTrain(mean1, mean2)

mean3 = np.array([3, 0])
mean4 = np.array([0, 3])
PrimalTrain(mean3, mean4)

