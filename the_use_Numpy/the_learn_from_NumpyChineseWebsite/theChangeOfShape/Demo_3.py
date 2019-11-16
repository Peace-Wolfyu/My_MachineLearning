# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 14:41'


"""

3、数组合并
concatenate

沿着一条轴连接一组（多个）数组。除了与axis对应的轴之外，其它轴必须有相同的形状。

vstack、row_stack

以追加行的方式对数组进行连接（沿轴0）

hstack

以追加列的方式对数组进行连接（沿轴1）

column_stack

类似于hstack，但是会先将一维数组转换为二维列向量

dstack

以面向“深度”的方式对数组进行叠加

"""

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

# print(a)
# print("")
# print(b)
# >>>
# [[1 2]
#  [3 4]]
#
# [[5 6]]

# 可以查看函数说明
# print(np.concatenate((a, b), axis=0))

# >>>
# [[1 2]
#  [3 4]
#  [5 6]]

# print(b.T)
# >>>
# [[5]
#  [6]]

# print(a.shape)
# print(b.T.shape)
# >>>
# (2, 2)
# (2, 1)

# print(np.concatenate((a, a.T), axis=1))
# >>>
# [[1 2 1 3]
#  [3 4 2 4]]


aa = np.array([[1, 2, 3],
              [4, 5, 6]])

bb = np.array([[10, 20, 30],
              [40, 50, 60]])

# print(aa)
# print(bb)
# >>>

# [[1 2 3]
#  [4 5 6]]

# [[10 20 30]
#  [40 50 60]]

# print(np.vstack((aa, bb)))
# >>>
# [[ 1  2  3]
#  [ 4  5  6]
#  [10 20 30]
#  [40 50 60]]

# print(np.row_stack((aa , bb)))
# >>>
# [[ 1  2  3]
#  [ 4  5  6]
#  [10 20 30]
#  [40 50 60]]


# print(np.hstack((aa, bb)))
# >>>
# [[ 1  2  3 10 20 30]
#  [ 4  5  6 40 50 60]]

# print(np.column_stack((aa, bb)))
# >>>
# [[ 1  2  3 10 20 30]
#  [ 4  5  6 40 50 60]]

# print(np.dstack((aa, bb)))
# >>>
# [[[ 1 10]
#   [ 2 20]
#   [ 3 30]]
#
#  [[ 4 40]
#   [ 5 50]
#   [ 6 60]]]






















