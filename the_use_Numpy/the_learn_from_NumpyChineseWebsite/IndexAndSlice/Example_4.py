# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 14:07'

import numpy as np


"""
布尔索引
"""
cities = np.array(['bj', 'cd', 'sh', 'gz', 'cd'])

# print(cities)
# >>>
# ['bj' 'cd' 'sh' 'gz' 'cd']

data = np.arange(20).reshape((5,4))
# print(data)
# >>>
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]
#  [16 17 18 19]]

# print(cities == 'cd')
#
# >>>
# [False  True False False  True]

# print(data[cities == 'cd'])
# >>>
# [[ 4  5  6  7]
#  [16 17 18 19]]

# print(data[cities == 'cd', :1])
# >>>
# [[ 4]
#  [16]]

# print(cities != 'cd')
# >>>
# [ True False  True  True False]

# print(data[cities != 'cd'])  # 等效于 data[~(cities == 'cd')]
# >>>
# [[ 0  1  2  3]
#  [ 8  9 10 11]
#  [12 13 14 15]]






