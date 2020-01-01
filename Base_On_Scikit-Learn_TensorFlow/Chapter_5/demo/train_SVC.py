# -*- coding: utf-8 -*-
# @Time  : 2020/1/1 19:28
# @Author : Mr.Lin


from sklearn.svm import SVC

# SVM Classifier model
from Chapter_5.demo.iris_data import X, y

svm_clf = SVC(kernel="linear", C=float("inf"))
svm_clf.fit(X, y)
