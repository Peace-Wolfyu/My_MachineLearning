# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 13:43'



import numpy as np

# 浅拷贝


a = np.arange(12)
c = a.view()

# print(a)
# print("")
# print(c)
# >>>
# [ 0  1  2  3  4  5  6  7  8  9 10 11]
#
# [ 0  1  2  3  4  5  6  7  8  9 10 11]

# print(c is a)

# >>>False

# print(c.base is a)
# >>>True

c[2] = 25656
# print(a)
# print("")
# print(c)
# >>>
# [    0     1 25656     3     4     5     6     7     8     9    10    11]
#
# [    0     1 25656     3     4     5     6     7     8     9    10    11]


c.shape = (12,1)

# print(a)
# # print("")
# # print(c)

# >>>
# [    0     1 25656     3     4     5     6     7     8     9    10    11]
#
# [[    0]
#  [    1]
#  [25656]
#  [    3]
#  [    4]
#  [    5]
#  [    6]
#  [    7]
#  [    8]
#  [    9]
#  [   10]
#  [   11]]






































