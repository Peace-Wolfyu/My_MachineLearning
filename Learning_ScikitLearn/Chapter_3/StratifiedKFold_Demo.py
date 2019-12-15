# -*- coding: utf-8 -*-
# @Time  : 2019/12/15 20:16
# @Author : Mr.Lin

'''

分层 k 折
StratifiedKFold 是 k-fold 的变种，会返回 stratified（分层） 的折叠：每个小集合中， 各个类别的样例比例大致和完整数据集中相同。

在有 10 个样例的，有两个略不均衡类别的数据集上进行分层 3-fold 交叉验证的例子:
'''


from sklearn.model_selection import StratifiedKFold
import numpy as np


X = np.ones(10)
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
skf = StratifiedKFold(n_splits=3)

# [2 3 6 7 8 9]  [0 1 4 5]
# [0 1 3 4 5 8 9]  [2 6 7]
# [0 1 2 4 5 6 7]  [3 8 9]


for train, test in skf.split(X, y):
     print("%s  %s" % (train, test))


#   ########################################
print("")
# Examples

import numpy as np
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=2)

'''
StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
TRAIN: [1 3] TEST: [0 2]
TRAIN: [0 2] TEST: [1 3]
'''
skf.get_n_splits(X, y)
print(skf)  # doctest: +NORMALIZE_WHITESPACE

for train_index, test_index in skf.split(X, y):
     print("TRAIN:", train_index, "TEST:", test_index)
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]

