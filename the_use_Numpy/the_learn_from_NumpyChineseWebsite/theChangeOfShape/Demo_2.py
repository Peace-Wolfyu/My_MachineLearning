# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 14:42'


"""
2、扁平化处理
"""


import numpy as np

ndarr = np.arange(12).reshape(3, 4)

# print(ndarr)
#
# >>>
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# order='C'，按列

# print(ndarr.flatten() )
# >>>
# [ 0  1  2  3  4  5  6  7  8  9 10 11]

 # 按行
# print(ndarr.flatten(order='F'))
# >>>
# [ 0  4  8  1  5  9  2  6 10  3  7 11]

ravel_ndarr = ndarr.ravel() # flatten()返回新对象，ravel()返回视图

# print(ravel_ndarr)
# >>>
# [ 0  1  2  3  4  5  6  7  8  9 10 11]

ravel_ndarr[1] = 100
# print(ndarr)
# >>>
# [[  0 100   2   3]
#  [  4   5   6   7]
#  [  8   9  10  11]]


































































