# -*- coding: utf-8 -*-
# @Time  : 2019/11/15 20:47
# @Author : Mr.Lin


import numpy as np

"""
和MATLAB不同，*是元素逐个相乘，而不是矩阵乘法。在Numpy中使用dot来进行矩阵乘法：

"""


x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))
print("")
# >>>
# 219
# 219

# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))
print("")
# >>>
# [29 67]
# [29 67]
# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))
print("")
# >>>
# [[19 22]
#  [43 50]]
# [[19 22]
#  [43 50]]


































