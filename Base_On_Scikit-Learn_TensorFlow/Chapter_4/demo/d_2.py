# -*- coding: utf-8 -*-
# @Time  : 2019/12/26 14:04
# @Author : Mr.Lin
"""

用随机梯度下降

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
# 随机生成一些线性数据

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# 实际用来生成数据的函数是y＝4+3x 0 +高斯噪声
print(theta_best)
print("----------------------------------------------")


lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_)
print(lin_reg.coef_)
print("----------------------------------------------")


eta = 0.1
n_iterations = 1000
m = 100
theta = np.random.randn(2, 1)

# 用随机梯度下降
for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print(theta)
print("----------------------------------------------")
