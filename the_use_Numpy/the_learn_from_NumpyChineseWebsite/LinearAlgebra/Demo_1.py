# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 16:01'



"""

常用的函数

diag(v, k=0)

以一维数组的形式返回方阵的对角线（或非对角线）元素，或将一维数组转换为方阵（非对角线元素为0）

dot(a, b[, out])

两个数组的点积

trace(a[, offset, axis1, axis2, dtype, out])

计算对角线元素的和

linalg.det(a)

计算矩阵行列式

linalg.inv(a)

计算方阵的逆
"""

import numpy as np

arr = np.arange(9).reshape((3,3))

# print(arr)
# >>>
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]

# print(np.diag(arr)
# )
# >>>
# [0 4 8]

#
# print(np.diag(arr, 1)
# )
# >>>
# [1 5]

diag_arr = np.array([1, 2, 3])
# print(np.diag(diag_arr)
# )
# >>>
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]

 # 计算矩阵乘积
a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]

# print(np.dot(a, b)
# )
# >>>
# [[4 1]
#  [2 2]]

# print(# 计算对角线元素的和
# # np.trace(arr))
# # >>>12

# 计算矩阵行列式
arr2 = np.array([[1, 2], [3, 4]])
# print(np.linalg.det(arr2))
# >>>
# -2.0000000000000004


# 计算矩阵的逆
# print(np.linalg.inv(arr2))
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# print(np.linalg.inv(arr2).dot(arr2))