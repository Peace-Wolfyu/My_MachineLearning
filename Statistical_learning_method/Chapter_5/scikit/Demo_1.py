# -*- coding: utf-8 -*-
# @Time  : 2019/12/9 19:59
# @Author : Mr.Lin
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
import matplotlib.pyplot as plt
import numpy as np
"""
使用预剪枝
"""


cancer = load_breast_cancer()


X_train,X_test,y_train,y_test = train_test_split(
    cancer.data,cancer.target,stratify=cancer.target,random_state=42
)

tree = DecisionTreeClassifier(random_state=0)

tree.fit(X_train,y_train)

print("Accuracy {: .3f}".format(tree.score(X_train,y_train)))
print("")
print("Accuracy {: .3f}".format(tree.score(X_test,y_test)))


'''
训练集精度为 100

因为叶子节点都是纯的

足以完美记住训练数据的所有标签

未剪枝的树容易过拟合
'''
# Accuracy  1.000
#
# Accuracy  0.937


'''
进行预剪枝

设置 max_depth = 4

意味着只可以连续问四个问题
'''
print("")
tree_1 = DecisionTreeClassifier(max_depth=4,random_state= 0 )

tree_1.fit(X_train,y_train)

print("Accuracy {: .3f}".format(tree_1.score(X_train,y_train)))
print("")
print("Accuracy {: .3f}".format(tree_1.score(X_test,y_test)))

# Accuracy  0.988
#
# Accuracy  0.951


export_graphviz(tree,out_file="tree.dot",class_names=["malignant","benign"],
                feature_names=cancer.feature_names,impurity=False,filled=True)


'''
树的特征重要性
'''


# 每个特征对树的重要性进行排序

print("Feature importances {}".format(tree.feature_importances_))

# Feature importances [0.         0.00752597 0.         0.         0.00903116 0.
#  0.00752597 0.         0.         0.         0.00975731 0.04630969
#  0.         0.00238745 0.00231135 0.         0.         0.
#  0.         0.00668975 0.69546322 0.05383211 0.         0.01354675
#  0.         0.         0.01740312 0.11684357 0.01137258 0.        ]




def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()

plot_feature_importances_cancer(tree)

















