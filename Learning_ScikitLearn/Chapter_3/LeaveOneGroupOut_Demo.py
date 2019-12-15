# -*- coding: utf-8 -*-
# @Time  : 2019/12/15 20:25
# @Author : Mr.Lin

'''

LeaveOneGroupOut 是一个交叉验证方案，它根据第三方提供的 array of integer groups （整数组的数组）来提供样本。这个组信息可以用来编码任意域特定的预定义交叉验证折叠。

每个训练集都是由除特定组别以外的所有样本构成的。

例如，在多个实验的情况下， LeaveOneGroupOut 可以用来根据不同的实验创建一个交叉验证：我们使用除去一个实验的所有实验的样本创建一个训练集:

'''





from sklearn.model_selection import LeaveOneGroupOut

X = [1, 5, 10, 50, 60, 70, 80]
y = [0, 1, 1, 2, 2, 2, 2]
groups = [1, 1, 2, 2, 3, 3, 3]
logo = LeaveOneGroupOut()

# [2 3 4 5 6]  [0 1]
# [0 1 4 5 6]  [2 3]
# [0 1 2 3]  [4 5 6]
for train, test in logo.split(X, y, groups=groups):
      print("%s  %s" % (train, test))


#   #############################
print("")
# Examples

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 1, 2])
groups = np.array([1, 1, 2, 2])
logo = LeaveOneGroupOut()
logo.get_n_splits(X, y, groups)

print(logo.get_n_splits(X, y, groups)
)
logo.get_n_splits(groups=groups)  # 'groups' is always required

print(logo.get_n_splits(groups=groups))
print(logo)

for train_index, test_index in logo.split(X, y, groups):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_train, X_test, y_train, y_test)
# 2
# 2
# LeaveOneGroupOut()
# TRAIN: [2 3] TEST: [0 1]
# [[5 6]
#  [7 8]] [[1 2]
#  [3 4]] [1 2] [1 2]
# TRAIN: [0 1] TEST: [2 3]
# [[1 2]
#  [3 4]] [[5 6]
#  [7 8]] [1 2] [1 2]














































































