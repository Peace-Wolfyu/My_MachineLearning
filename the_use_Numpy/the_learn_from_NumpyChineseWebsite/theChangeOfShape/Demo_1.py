# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 14:30'



"""
数组形状变换
重塑
扁平化处理
数组合并
数组拆分
数组扩充
数组转置和轴对换
"""

"""
1、重塑
ndarray.reshape(shape, order='C')
ndarray.resize()
reshape 函数返回修改后的新对象，而 ndarray.resize 方法修改数组本身

重塑的各个维度上整数的乘积必须等于arr.size
如果想让自动计算某个轴上的大小，可以传入-1
"""


import numpy as np

arr = np.arange(12)
arr2 = arr.copy()
arr.reshape((4, 3)) # order默认为‘C’ ，按列读取。等效于reshape(4, 3)

# print(arr)
# print("")
# print(arr2)
# >>>
# [ 0  1  2  3  4  5  6  7  8  9 10 11]
#
# [ 0  1  2  3  4  5  6  7  8  9 10 11]

# print(arr.reshape((4, 3)))
# >>>
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

# resize()会修改对象本身
arr2.resize((4,3))
# print(arr2)
# >>>
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]


"""
多维数组重塑
"""
ndarr = arr.reshape(4, 3)
# print(ndarr)
# >>>
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]



# print(ndarr.reshape(2, 6))
#
# >>>
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]]


# 重塑为三维度
# print(arr.reshape((2,2,3)))
# >>>
# [[[ 0  1  2]
#   [ 3  4  5]]
#
#  [[ 6  7  8]
#   [ 9 10 11]]]


# print(arr.reshape((3, 2, -1)))
# >>>
# [[[ 0  1]
#   [ 2  3]]
#
#  [[ 4  5]
#   [ 6  7]]
#
#  [[ 8  9]
#   [10 11]]]





























































