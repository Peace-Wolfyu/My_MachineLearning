# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 15:02'



"""
5、repeat和tile

"""

import numpy as np


arr = np.arange(4)

# print(arr)
# >>>
# [0 1 2 3]


# print(np.repeat(arr, 2))
# >>>
# [0 0 1 1 2 2 3 3]


# 指定每个元素的重复次数
# 0重复2次，1重复3次，2重复4次，3重复5次
# print(np.repeat(arr, [2, 3, 4, 5]))
# >>>
# [0 0 1 1 1 2 2 2 2 3 3 3 3 3]

"""
多维数组repeat

"""
ndarr = np.arange(6).reshape(2, 3)
# print(ndarr)
# >>>
# [[0 1 2]
#  [3 4 5]]

# print(np.repeat(ndarr, 2) )# 不指定轴会被扁平化
# >>>
# [0 0 1 1 2 2 3 3 4 4 5 5]


# print(np.repeat(ndarr, 2, axis=0))
# >>>
# [[0 1 2]
#  [0 1 2]
#  [3 4 5]
#  [3 4 5]]


# print(np.repeat(ndarr, 2, axis=1))
# >>>
# [[0 0 1 1 2 2]
#  [3 3 4 4 5 5]]

"""
tile ———— 针对整个数组
"""

# print(ndarr)
# >>>
# [[0 1 2]
#  [3 4 5]]

# print(np.tile(ndarr, 2)) # 对标量是横向扩展)
# >>>
# [[0 1 2 0 1 2]
#  [3 4 5 3 4 5]]

# print(np.tile(ndarr, (1,2)))
# >>>
# [[0 1 2 0 1 2]
#  [3 4 5 3 4 5]]

# print(np.tile(ndarr, (2,3)))
# >>>
# [[0 1 2 0 1 2 0 1 2]
#  [3 4 5 3 4 5 3 4 5]
#  [0 1 2 0 1 2 0 1 2]
#  [3 4 5 3 4 5 3 4 5]]







