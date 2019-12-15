# -*- coding: utf-8 -*-
# @Time  : 2019/12/15 20:08
# @Author : Mr.Lin


'''


留 P 交叉验证 (LPO)
LeavePOut 与 LeaveOneOut 非常相似，因为它通过从整个集合中删除 p 个样本来创建所有可能的 训练/测试集。对于 n 个样本，这产生了 {n \choose p} 个 训练-测试 对。与 LeaveOneOut 和 KFold 不同，当 p > 1 时，测试集会重叠。

在有 4 个样例的数据集上使用 Leave-2-Out 的例子:

'''


from sklearn.model_selection import LeavePOut
import numpy as np
X = np.ones(4)
lpo = LeavePOut(p=2)


# [2 3]  [0 1]
# [1 3]  [0 2]
# [1 2]  [0 3]
# [0 3]  [1 2]
# [0 2]  [1 3]
# [0 1]  [2 3]
for train, test in lpo.split(X):
    print("%s  %s" % (train, test))