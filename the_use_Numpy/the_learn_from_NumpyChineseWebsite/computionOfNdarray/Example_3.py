# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 13:34'


import numpy as np
"""
NumPy数组计算

"""


# 1、数组与标量的算术运算

data = np.arange(10).reshape((5,2))
# print(data)
# >>>
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]
#  [8 9]]

# print(data+data)
# >>>
# [[ 0  2]
#  [ 4  6]
#  [ 8 10]
#  [12 14]
#  [16 18]]

# print(data*data)

# [[ 0  1]
#  [ 4  9]
#  [16 25]
#  [36 49]
#  [64 81]]

# print(data+1)
# >>>
# [[ 1  2]
#  [ 3  4]
#  [ 5  6]
#  [ 7  8]
#  [ 9 10]]

data2 = data + 1

# shape相同的数组之间比较会生成布尔数组
# print(data2 > data)
# >>>
# [[ True  True]
#  [ True  True]
#  [ True  True]
#  [ True  True]
#  [ True  True]]


# 不同类型的数组计算之后的结果的类型会向上转换为更精确的类型（如int类型的数组a和float类型的数组b求和得到c的类型为float，而不是int）
int_arr = np.array([1, 2, 3])
float_arr = np.array([0.1, 0.2, 0.3])
res_arr = int_arr + float_arr
print(res_arr.dtype)
# >>>
# float64
































