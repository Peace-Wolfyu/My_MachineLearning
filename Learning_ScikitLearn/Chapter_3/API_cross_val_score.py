# -*- coding: utf-8 -*-
# @Time  : 2019/12/15 19:42
# @Author : Mr.Lin



from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()

#  [0.33150734 0.08022311 0.03531764]
print(cross_val_score(lasso, X, y, cv=3))  # doctest: +ELLIPSIS


