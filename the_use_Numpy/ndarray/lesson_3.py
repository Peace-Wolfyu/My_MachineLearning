# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/12 17:15'

"""
数组转置和咒对换
"""

import numpy as  np


arr = np.arange(15).reshape((3,5))

print(arr)


"""
转置操作
"""
print(arr.T)



arr_1 = np.random.rand(6,3)

print(arr_1)

print(np.dot(arr_1.T,arr_1))