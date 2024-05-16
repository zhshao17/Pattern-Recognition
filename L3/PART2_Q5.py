import numpy as np
import math
import matplotlib.pyplot as plt
import random


def pic(x1, x2):
    x = np.arange(x1, x2, 0.01)
    y = np.array([])
    for i in range(len(x)):
        y = np.append(y, x[i] * math.cos(0.25 * math.pi * x[i]))
    plt.plot(x, y)


def arrow(x, f):
    for i in range(len(x) - 1):
        plt.arrow(x[i], f[i], x[i + 1] - x[i], f[i + 1] - f[i],
                  linestyle='-', color='r', linewidth='2', head_width=0.1, head_length=0.1)


def Grad(epochs, x0, learning_rate):
    x = x0
    x_record = np.array([x0])
    f = np.array([x * math.cos(0.25 * math.pi * x)])
    for i in range(epochs):
        grad = math.cos(0.25 * math.pi * x) + x * -1 * math.sin(0.25 * math.pi * x) * 0.25 * math.pi
        x = x - learning_rate * grad
        if abs(grad) < 1e-10:
            break
        x_record = np.append(x_record, x)
        f = np.append(f, x * math.cos(0.25 * math.pi * x))
    print('x:', x_record)
    print('f(x):', f)
    plt.figure(1)
    pic(np.min(x_record) - 0.5, np.max(x_record) + 0.5)
    plt.scatter(x_record, f, marker='o', color='g')
    arrow(x_record, f)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Grad')
    # plt.savefig("Grad1.png")
    plt.show()


def SGD(epochs, x0, learning_rate):
    x = x0
    x_record = np.array([x0])
    f = np.array([x * math.cos(0.25 * math.pi * x)])
    for i in range(epochs):
        grad = math.cos(0.25 * math.pi * x) + x * -1 * math.sin(0.25 * math.pi * x) * 0.25 * math.pi
        x = x - learning_rate * grad + random.gauss(mu=0, sigma=0.1)
        if abs(grad) < 1e-10:
            break
        x_record = np.append(x_record, x)
        f = np.append(f, x * math.cos(0.25 * math.pi * x))
    print('x:', x_record)
    print('f(x):', f)
    plt.figure(1)
    pic(np.min(x_record) - 0.5, np.max(x_record) + 0.5)
    plt.scatter(x_record, f, marker='o', color='g')
    arrow(x_record, f)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('SGD')
    # plt.savefig("SGD1.png")
    plt.show()


def Adagrad(epochs, x0, learning_rate, epsilon):
    x = x0
    x_record = np.array([x0])
    grad_record = np.array([])
    f = np.array([x * math.cos(0.25 * math.pi * x)])
    for i in range(epochs):
        # batch = 1
        grad = math.cos(0.25 * math.pi * x) + x * -1 * math.sin(0.25 * math.pi * x) * 0.25 * math.pi
        grad_record = np.append(grad_record, grad)
        sigma = math.sqrt(np.sum(grad_record ** 2) / len(grad_record)) + epsilon
        x = x - learning_rate / sigma * grad
        if abs(grad) < 1e-10:
            break
        x_record = np.append(x_record, x)
        f = np.append(f, x * math.cos(0.25 * math.pi * x))
    print('x:', x_record)
    print('f(x):', f)
    pic(np.min(x_record) - 0.5, np.max(x_record) + 0.5)
    plt.scatter(x_record, f, marker='o', color='g')
    arrow(x_record, f)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Adagrad')
    # plt.savefig("Adagrad1.png")
    plt.show()


def RMSProp(epochs, x0, learning_rate, alpha, epsilon):
    x = x0
    x_record = np.array([x0])
    grad_record = np.array([])
    sigma_record = np.array([])
    f = np.array([x * math.cos(0.25 * math.pi * x)])
    for i in range(epochs):
        # batch = 1
        grad = math.cos(0.25 * math.pi * x) + x * -1 * math.sin(0.25 * math.pi * x) * 0.25 * math.pi
        grad_record = np.append(grad_record, grad)
        if i == 0:
            sigma = math.sqrt(grad ** 2) + epsilon
        else:
            sigma = math.sqrt(alpha * sigma_record[i - 1] ** 2 + (1 - alpha) * grad_record[i] ** 2) + epsilon
        # sigma = math.sqrt(np.sum(grad_record ** 2) / len(grad_record)) + epsilon
        x = x - (learning_rate / sigma) * grad
        if abs(grad) < 1e-10:
            break
        x_record = np.append(x_record, x)
        sigma_record = np.append(sigma_record, sigma)
        f = np.append(f, x * math.cos(0.25 * math.pi * x))
    print('x:', x_record)
    print('f(x):', f)
    pic(np.min(x_record) - 0.5, np.max(x_record) + 0.5)
    arrow(x_record, f)
    plt.scatter(x_record, f, marker='o', color='g')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('RMProp')
    # plt.savefig("RMProp1.png")
    plt.show()


def Momentum(epochs, x0, learning_rate, lambda1):
    x = x0
    m = 0
    x_record = np.array([x0])
    f = np.array([x * math.cos(0.25 * math.pi * x)])
    for i in range(epochs):
        # batch = 1
        grad = math.cos(0.25 * math.pi * x) + x * -1 * math.sin(0.25 * math.pi * x) * 0.25 * math.pi
        m = lambda1 * m - learning_rate * grad
        x = x + m
        if abs(grad) < 1e-100:
            break
        x_record = np.append(x_record, x)
        f = np.append(f, x * math.cos(0.25 * math.pi * x))
    print('x:', x_record)
    print('f(x):', f)
    pic(np.min(x_record) - 0.5, np.max(x_record) + 0.5)
    plt.scatter(x_record, f, marker='o', color='g')
    arrow(x_record, f)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Momentum')
    # plt.savefig("Momentum1.png")
    plt.show()


def Adam(epochs, x0, learning_rate, alpha, beta1, beta2, epsilon):
    x = x0
    m = 0
    v = 0
    x_record = np.array([x0])
    f = np.array([x * math.cos(0.25 * math.pi * x)])
    for i in range(epochs):
        # batch = 1
        grad = math.cos(0.25 * math.pi * x) + x * -1 * math.sin(0.25 * math.pi * x) * 0.25 * math.pi
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m1 = m / (1 - beta1 ** (i + 1))
        v1 = v / (1 - beta2 ** (i + 1))
        x = x - learning_rate * m1 / (math.sqrt(v1) + epsilon)
        if abs(grad) < 1e-100:
            break
        x_record = np.append(x_record, x)
        f = np.append(f, x * math.cos(0.25 * math.pi * x))
    print('x:', x_record)
    print('f(x):', f)
    pic(np.min(x_record) - 0.5, np.max(x_record) + 0.5)
    plt.scatter(x_record, f, marker='o', color='g')
    arrow(x_record, f)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Adam')
    # plt.savefig("Adam1.png")
    plt.show()


# Grad(epochs=40, x0=-4, learning_rate=0.4)
# # SGD(epochs=10, x0=-4, learning_rate=0.4)
# # Adagrad(epochs=50, x0=-4, learning_rate=0.4, epsilon=1e-6)
# # RMSProp(epochs=50, x0=-4, learning_rate=0.4, alpha=0.9, epsilon=1e-6)
# # Momentum(epochs=50, x0=-4, learning_rate=0.4, lambda1=0.9)
Adam(epochs=10, x0=-4, learning_rate=0.4, alpha=0.9, beta1=0.99, beta2=0.999, epsilon=1e-6)
