# -*- coding: utf-8 -*-
# @Time  : 2019/12/26 14:32
# @Author : Mr.Lin

"""
随机梯度下降的前10步
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 随机生成一些线性数据

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance

n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters


def learning_schedule(t):
    return t0 / (t + t1)


theta = np.random.randn(2, 1)  # random initialization
m = len(X_b)
for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:  # not shown in the book
            y_predict = X_new_b.dot(theta)  # not shown
            style = "b-" if i > 0 else "r--"  # not shown
            plt.plot(X_new, y_predict, style)  # not shown
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        # theta_path_sgd.append(theta)                 # not shown

plt.plot(X, y, "b.")  # not shown
plt.xlabel("$x_1$", fontsize=18)  # not shown
plt.ylabel("$y$", rotation=0, fontsize=18)  # not shown
plt.axis([0, 2, 0, 15])  # not shown
plt.show()
