# -*- coding: utf-8 -*-
# @Time  : 2019/11/12 20:03
# @Author : Mr.Lin

"""
排序
"""

import numpy as np

arr = np.random.randn(6)

print(arr)

arr.sort()

print(arr)

"""
多维数组可以在任何一个轴向上进行排序，只需将轴编号传给sort即可
"""

arr_1 = np.random.randn(5,3)

print(arr_1)

arr_1.sort(1)

print(arr_1)