# -*- coding: utf-8 -*-
# @Time  : 2019/12/26 15:22
# @Author : Mr.Lin

"""

使用Scikit-Learn执行闭式解的岭回归

"""
import numpy as np
from sklearn.linear_model import Ridge, SGDRegressor

np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)

ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(X, y)
print(ridge_reg.predict([[1.5]]))
print("----------------------------------------------")

# 使用随机梯度下降：

# 超参数penalty设置的是使用正则项的类型。设为"l2"表示希望
# SGD在成本函数中添加一个正则项，等于权重向量的l 2 范数的平方的
# 一半，即岭回归
sgd_reg = SGDRegressor(penalty="l2", random_state=42)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[1.5]]))



















