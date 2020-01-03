# -*- coding: utf-8 -*-
# @Time  : 2020/1/3 15:00
# @Author : Mr.Lin
"""

鸢尾花数据集

"""
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, 2:]  # petal length and width
y = iris.target
