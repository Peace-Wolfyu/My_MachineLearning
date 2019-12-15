# -*- coding: utf-8 -*-
# @Time  : 2019/12/15 20:09
# @Author : Mr.Lin


'''

随机排列交叉验证 a.k.a. Shuffle & Split
ShuffleSplit

ShuffleSplit 迭代器 将会生成一个用户给定数量的独立的训练/测试数据划分。样例首先被打散然后划分为一对训练测试集合。

可以通过设定明确的 random_state ，使得伪随机生成器的结果可以重复。

'''


from sklearn.model_selection import ShuffleSplit, cross_val_score
import numpy as np
from Learning_ScikitLearn.Chapter_3.Data import iris,logreg


X = np.arange(5)
ss = ShuffleSplit(n_splits=3, test_size=0.25,
random_state=0)

# [1 3 4]  [2 0]
# [1 4 3]  [0 2]
# [4 0 2]  [1 3]
for train_index, test_index in ss.split(X):
     print("%s  %s" % (train_index, test_index))

# ShuffleSplit 可以替代 KFold 交叉验证，因为其提供了细致的训练 / 测试划分的 数量和样例所占的比例等的控制。
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)

# Cross-validation scores:
# [0.98666667 0.93333333 0.93333333 0.94666667 0.93333333 0.92
#  0.92       0.90666667 0.96       0.90666667]

print("Cross-validation scores:\n{}".format(scores))


'''
打乱划分交叉验证可以在训练集和测试集大小之外独立控制迭代次数，这有时是很有帮助
的。它还允许在每次迭代中仅使用部分数据，这可以通过设置 train_size 与 test_size 之
和不等于 1 来实现。用这种方法对数据进行二次采样可能对大型数据上的试验很有用。
ShuffleSplit 还有一种分层的形式，其名称为 StratifiedShuffleSplit ，它可以为分类任
务提供更可靠的结果。
'''