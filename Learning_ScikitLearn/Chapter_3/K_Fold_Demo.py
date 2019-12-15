# -*- coding: utf-8 -*-
# @Time  : 2019/12/15 19:53
# @Author : Mr.Lin


'''

K 折
KFold 将所有的样例划分为 k 个组，称为折叠 (fold) （如果 k = n， 这等价于 Leave One Out（留一） 策略），都具有相同的大小（如果可能）。预测函数学习时使用 k - 1 个折叠中的数据，最后一个剩下的折叠会用于测试。

在 4 个样例的数据集上使用 2-fold 交叉验证的例子:

'''


import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from Learning_ScikitLearn.Chapter_3.Data import iris,logreg
X = ["a", "b", "c", "d"]

'''
 scikit-learn
允许提供一个交叉验证分离器（cross-validation splitter）作为 cv 参数，来对数据划分过程
进行更精细的控制。对于大多数使用场景而言，回归问题默认的 k 折交叉验证与分类问题
的分层 k 折交叉验证的表现都很好，但有些情况下你可能希望使用不同的策略。比如说，
我们想要在一个分类数据集上使用标准 k 折交叉验证来重现别人的结果。为了实现这一
点，我们首先必须从 model_selection 模块中导入 KFold 分离器类
'''
kf = KFold(n_splits=2)

# [2 3]  [0 1]
# [0 1]  [2 3]
# 每个折叠由两个 arrays 组成，第一个作为 training set ，另一个作为 test set 。 由此，可以通过使用 numpy 的索引创建训练/测试集合:
for train, test in kf.split(X):
    print("%s  %s" % (train, test))


# Cross-validation scores:
# [0.28       0.33333333]

print("Cross-validation scores:\n{}".format(
cross_val_score(logreg, iris.data, iris.target, cv=kf)))