# -*- coding: utf-8 -*-
# @Time  : 2020/1/3 15:01
# @Author : Mr.Lin


from sklearn.tree import export_graphviz, DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from Chapter_6.demo.iris_data import X, y, iris
import matplotlib.pyplot as plt
import numpy as np


tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)

export_graphviz(
    tree_clf,
    out_file="iris_tree.dot",
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)



