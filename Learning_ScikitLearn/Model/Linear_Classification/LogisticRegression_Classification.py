# -*- coding: utf-8 -*-

# @Time  : 2019/12/16 15:55

# @Author : Mr.Lin


'''

用于分类的线性模型
线性模型也广泛应用于分类问题。我们首先来看二分类。这时可以利用下面的公式进行
预测：
ŷ = w[0] * x[0] + w[1] * x[1] + …+ w[p] * x[p] + b > 0
这个公式看起来与线性回归的公式非常相似，但我们没有返回特征的加权求和，而是为预
测设置了阈值（0）。如果函数值小于 0，我们就预测类别 -1；如果函数值大于 0，我们就
预测类别 +1。对于所有用于分类的线性模型，这个预测规则都是通用的。同样，有很多种
不同的方法来找出系数（w）和截距（b）。
对于用于回归的线性模型，输出 ŷ 是特征的线性函数，是直线、平面或超平面（对于更高
维的数据集）。对于用于分类的线性模型，决策边界是输入的线性函数。换句话说，（二
元）线性分类器是利用直线、平面或超平面来分开两个类别的分类器。本节我们将看到这
方面的例子。
学习线性模型有很多种算法。这些算法的区别在于以下两点：
• 系数和截距的特定组合对训练数据拟合好坏的度量方法；
• 是否使用正则化，以及使用哪种正则化方法。
不同的算法使用不同的方法来度量“对训练集拟合好坏”。由于数学上的技术原因，不可
能调节 w 和 b 使得算法产生的误分类数量最少。对于我们的目的，以及对于许多应用而
言，上面第一点（称为损失函数）的选择并不重要。
最常见的两种线性分类算法是 Logistic 回归（logistic regression）和线性支持向量机（linear
support vector machine，线性 SVM），前者在 linear_model.LogisticRegression 中实现，
后者在 svm.LinearSVC （SVC 代表支持向量分类器）中实现。虽然 LogisticRegression
的名字中含有回归（regression），但它是一种分类算法，并不是回归算法，不应与
LinearRegression 混淆。

'''
from sklearn.cross_validation import cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from Learning_ScikitLearn.Model.Linear_Classification.Data_Source import X_test,X_train,y_train,y_test,data_y,data_X
# logreg = LogisticRegression().fit(X_train, y_train)


# print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
# print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))


# Training set score: 0.955
# Test set score: 0.958
#
# [0.94827586 0.9137931  0.92982456 0.94736842 0.96491228 0.98245614
#  0.94736842 0.94642857 0.96428571 0.96428571]



# print("")
# print(cross_val_score(logreg, data_X, data_y, cv=10))

def test_C_Parameter():
    C = [0.1,1,10]

    for c in C:
        logreg = LogisticRegression(C=c)
        logreg.fit(X_train,y_train)
        print("C为：{}下的分数：{}\n".format(c,cross_val_score(logreg, data_X, data_y, cv=10)))


test_C_Parameter()

















