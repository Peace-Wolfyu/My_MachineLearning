# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 13:59'


import numpy as np

"""
多维切片

"""
ndarr = np.arange(16).reshape((4, 4))

# print(ndarr)
#
# >>>
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]]

# print(ndarr[:2])# 行，等效于arr2[:2, :]
# >>>
# [[0 1 2 3]
#  [4 5 6 7]]

# print(ndarr[:2,:2])
# >>>
# [[0 1]
#  [4 5]]


# print(ndarr[::2, -2] )# 步长为2的行的倒数第二列
# >>>
# [ 2 10]














