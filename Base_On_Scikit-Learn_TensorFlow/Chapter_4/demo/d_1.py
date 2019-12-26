# -*- coding: utf-8 -*-
# @Time  : 2019/12/26 13:07
# @Author : Mr.Lin

"""

学习线性模型的标准方程

"""

import matplotlib.pyplot as plt
import numpy as np

# 随机生成一些线性数据

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# plt.plot(X, y, "b.")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.axis([0, 2, 0, 15])
# 画出散点图
# plt.show()

# 标准方程来计算

X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# 实际用来生成数据的函数是y＝4+3x 0 +高斯噪声
print(theta_best)
print("----------------------------------------------")
# 预测

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
print(y_predict)
print("----------------------------------------------")

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
# plt.show()

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_)
print(lin_reg.coef_)















































































