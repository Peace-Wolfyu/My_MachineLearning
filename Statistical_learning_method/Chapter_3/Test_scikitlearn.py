# -*- coding: utf-8 -*-

# @Time  : 2019/12/2 15:05

# @Author : Mr.Lin

import math
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier





iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train, y_train)




print(clf_sk.score(X_test, y_test))





"""
sklearn.neighbors.KNeighborsClassifier
n_neighbors: 临近点个数
p: 距离度量
algorithm: 近邻算法，可选{'auto', 'ball_tree', 'kd_tree', 'brute'}
weights: 确定近邻的权重
"""















