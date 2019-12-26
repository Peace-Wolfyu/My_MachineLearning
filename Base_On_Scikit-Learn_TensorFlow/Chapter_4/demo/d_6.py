# -*- coding: utf-8 -*-
# @Time  : 2019/12/26 14:39
# @Author : Mr.Lin

"""
多项式回归
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 随机生成一个二次方程
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)


# 画出散点图
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
# plt.show()


# 直线永远不可能拟合这个数据

# 使用Scikit-Learn
# 的PolynomialFeatures类来对训练数据进行转换，将每个特征的平方（二次多项式）作为新特征加入训练集（这个例子中只有一个特
# 征）


poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print(X[0])
print("----------------------------------------------")
print(X_poly[0])
print("----------------------------------------------")

# X_poly现在包含原本的特征X和该特征的平方。现在对这个扩展
# 后的训练集匹配一个LinearRegression模型

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_)
print(lin_reg.coef_)











































