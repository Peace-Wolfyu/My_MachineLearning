# -*- coding: utf-8 -*-
# @Time  : 2019/12/2 20:18
# @Author : Mr.Lin


"""
测试 KNN

"""
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import Statistical_learning_method.Chapter_3.MyFunction as Myfun
from Statistical_learning_method.Chapter_3.DataSet_1 import data as source
# from Statistical_learning_method.Chapter_3.KNN import KNN as the_old_knn
# data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']


# data = np.array(df.iloc[:100, [0, 1, -1]])
data = source
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print("X_train:\n{}".format(X_train))
# print("")
# print("X_test : \n{}".format(X_test))
# print("")
# print("y_train:\n{}".format(y_train))
# print("")
# print("y_test:\n{}".format(y_test))


clf = Myfun.KNN_Model(X_train, y_train)
# clf = the_old_knn(X_train,y_train)

print(clf.score(X_test,y_test))


