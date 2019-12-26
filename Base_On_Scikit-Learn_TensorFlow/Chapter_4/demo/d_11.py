# -*- coding: utf-8 -*-
# @Time  : 2019/12/26 15:34
# @Author : Mr.Lin

"""

套索回归

"""
import numpy as np
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.linear_model import Lasso

np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)


lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
print(lasso_reg.predict([[1.5]]))

sgd_reg = SGDRegressor(penalty="l2", random_state=42)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[1.5]]))


