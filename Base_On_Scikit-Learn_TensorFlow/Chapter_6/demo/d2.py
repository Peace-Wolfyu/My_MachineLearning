# -*- coding: utf-8 -*-
# @Time  : 2020/1/3 15:17
# @Author : Mr.Lin

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from matplotlib.colors import ListedColormap

from Chapter_6.demo.d import plot_decision_boundary
from Chapter_6.demo.iris_data import X, y, iris



angle = np.pi / 180 * 20
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
Xr = X.dot(rotation_matrix)

tree_clf_r = DecisionTreeClassifier(random_state=42)
tree_clf_r.fit(Xr, y)

plt.figure(figsize=(8, 3))
plot_decision_boundary(tree_clf_r, Xr, y, axes=[0.5, 7.5, -1.0, 1], iris=False)

plt.show()