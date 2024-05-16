import numpy as np
import matplotlib.pyplot as plt
import Lecture2 as L2


def train_Q5():
    train_data = np.array([[0.2, 0.7],
                           [0.3, 0.3],
                           [0.4, 0.5],
                           [0.6, 0.5],
                           [0.1, 0.4],
                           [0.4, 0.6],
                           [0.6, 0.2],
                           [0.7, 0.4],
                           [0.8, 0.6],
                           [0.7, 0.5]])
    train_label = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    Wp_record, w_record = L2.Pocket(train_data, train_label, w_0=0., x_0=1., epochs=20)
    print('w:', w_record)
    print('Wp:', Wp_record)
    L2.plot(train_data, train_label, Wp_record[-1], title='Wp')
    L2.plot(train_data, train_label, w_record[-1], title='Wt+1')
    w = np.column_stack([Wp_record, w_record])


train_Q5()
