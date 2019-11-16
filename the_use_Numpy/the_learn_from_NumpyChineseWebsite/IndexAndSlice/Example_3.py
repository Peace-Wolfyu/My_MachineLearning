# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 14:02'


import numpy as np


"""
索引
"""
arr = np.arange(10)
ndarr = np.arange(16).reshape((4, 4))

# print(arr)
# print("")
# print(ndarr)

# >>>
# [0 1 2 3 4 5 6 7 8 9]
#
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]]

# print(arr[1])
# >>>1

# print(ndarr[1,1])
# >>>5

# print(ndarr[2,[1,2]]) # 第2行的第1、2列元素
# >>>[ 9 10]


# print(ndarr[:3, [0,1]] )# 前3行的第0、1列元素
# >>>
# [[0 1]
#  [4 5]
#  [8 9]]

























































































