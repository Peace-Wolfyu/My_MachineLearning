# -*- coding: utf-8 -*-
# @Time  : 2019/12/15 20:22
# @Author : Mr.Lin


'''

 k-fold
GroupKFold 是 k-fold 的变体，它确保同一个 group 在测试和训练集中都不被表示。 例如，如果数据是从不同的 subjects 获得的，每个 subject 有多个样本，并且如果模型足够灵活以高度人物指定的特征中学习，则可能无法推广到新的 subject 。 GroupKFold 可以检测到这种过拟合的情况
'''

from sklearn.model_selection import GroupKFold

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

gkf = GroupKFold(n_splits=3)

# [0 1 2 3 4 5]  [6 7 8 9]
# [0 1 2 6 7 8 9]  [3 4 5]
# [3 4 5 6 7 8 9]  [0 1 2]
for train, test in gkf.split(X, y, groups=groups):
     print("%s  %s" % (train, test))



# #################################################
# Examples
print("")
import numpy as np
from sklearn.model_selection import GroupKFold
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])
groups = np.array([0, 0, 2, 2])
group_kfold = GroupKFold(n_splits=2)


'''

2
GroupKFold(n_splits=2)
TRAIN: [0 1] TEST: [2 3]
[[1 2]
 [3 4]] [[5 6]
 [7 8]] [1 2] [3 4]
TRAIN: [2 3] TEST: [0 1]
[[5 6]
 [7 8]] [[1 2]
 [3 4]] [3 4] [1 2]
'''
print(group_kfold.get_n_splits(X, y, groups))

print(group_kfold)

for train_index, test_index in group_kfold.split(X, y, groups):
         print("TRAIN:", train_index, "TEST:", test_index)
         X_train, X_test = X[train_index], X[test_index]
         y_train, y_test = y[train_index], y[test_index]
         print(X_train, X_test, y_train, y_test)












































