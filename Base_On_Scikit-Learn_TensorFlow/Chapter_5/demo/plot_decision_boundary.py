# -*- coding: utf-8 -*-
# @Time  : 2020/1/1 19:31
# @Author : Mr.Lin

"""
左图显示了三种可能的线性分
类器的决策边界。其中虚线所代表的模型表现非常糟糕，甚至都无法
正确实现分类。其余两个模型在这个训练集上表现堪称完美，但是它
们的决策边界与实例过于接近，导致在面对新实例时，表现可能不会
太好。相比之下，右图中的实线代表SVM分类器的决策边界，这条
线不仅分离了两个类别，并且尽可能远离了最近的训练实例。你可以
将SVM分类器视为在类别之间拟合可能的最宽的街道（平行的虚线
所示）。因此这也叫作大间隔分类（large margin classification）

"""
import matplotlib.pyplot as plt
import numpy as np
from Chapter_5.demo.iris_data import X, y
from Chapter_5.demo.train_SVC import svm_clf


def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]
    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)


# Bad models
x0 = np.linspace(0, 5.5, 200)
# print(x0)
pred_1 = 5 * x0 - 20
pred_2 = x0 - 1.8
pred_3 = 0.1 * x0 + 0.5
plt.figure(figsize=(12, 2.7))
plt.subplot(121)
plt.plot(x0, pred_1, "g--", linewidth=2)
plt.plot(x0, pred_2, "m-", linewidth=2)
plt.plot(x0, pred_3, "r-", linewidth=2)
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris-Versicolor")
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="Iris-Setosa")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 5.5, 0, 2])
plt.subplot(122)
plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo")
plt.xlabel("Petal length", fontsize=14)
plt.axis([0, 5.5, 0, 2])
plt.show()
