# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 13:47'


import numpy as np

# 深拷贝
a = np.arange(12)
d = a.copy()
# print(a)
# print("")
# print(d)
# >>>
# [ 0  1  2  3  4  5  6  7  8  9 10 11]
#
# [ 0  1  2  3  4  5  6  7  8  9 10 11]

# print(d is a)
# >>>False

a[1] = 1048

# print(a)
# print("")
# print(d)
# >>>
# [   0 1048    2    3    4    5    6    7    8    9   10   11]
#
# [ 0  1  2  3  4  5  6  7  8  9 10 11]

d.shape = (4,3)
print(a)
print("")
print(d)
# >>>
# [   0 1048    2    3    4    5    6    7    8    9   10   11]
#
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]































