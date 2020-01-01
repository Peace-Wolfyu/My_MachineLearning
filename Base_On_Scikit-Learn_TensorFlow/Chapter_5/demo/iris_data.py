# -*- coding: utf-8 -*-
# @Time  : 2020/1/1 19:25
# @Author : Mr.Lin

"""

加载鸢尾花数据集

"""

from sklearn import datasets


iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
# print(X)
# print("")
y = y[setosa_or_versicolor]
# print(y)

