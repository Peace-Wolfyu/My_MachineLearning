# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 13:54'




import numpy as np


"""
索引和切片
数组切片是原始数组的视图，不是复制品，如果要得到切片的副本，应该使用copy()方法
在多维数组中索引，如果省略了后面的索引，则返回对象会是一个维度低一点的ndarray（它含有高一级维度上的所有数据）
1、切片
切片的基本语法 ndarray[i : j : k]，其中i为起始下标，j为结束下标（不包括j），k为步长（默认为1）
"""

data = np.arange(10)
# print(data)
# >>>
# [0 1 2 3 4 5 6 7 8 9]

# print(data[1:9:3])
# >>>
# [1 4 7]

# print(data[4:-3])
# >>>
# [4 5 6]

# print(data[9:4:-2])
# >>>
# [9 7 5]

# data = data[::2]
# print(data)
# >>>
# [0 2 4 6 8]

# data[:6] = 0
# print(data[6:])
# >>>
# [6 7 8 9]

# data[:] = 1
# print(data)
# >>>
# [1 1 1 1 1 1 1 1 1 1]










































































