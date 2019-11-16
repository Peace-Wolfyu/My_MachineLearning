# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 15:51'

"""
排序

sort(a[, axis, kind, order])

返回数组的排序副本

ndarray.sort([axis, kind, order])

就地对数组进行排序

lexsort(keys[, axis])

使用一系列键执行间接排序

argsort(a[, axis, kind, order])

返回对数组进行排序的索引

msort(a)

返回沿第一个轴排序的数组副本

sort_complex(a)

首先使用实部对复杂数组进行排序，然后使用虚部进行排序

partition(a, kth[, axis, kind, order])

返回数组的分区副本

argpartition(a, kth[, axis, kind, order])

使用kind关键字指定的算法沿给定轴执行间接分区
"""
import numpy as np

arr1 = np.random.randn(10)
print(arr1)
print("")
print("")
print(np.sort(arr1))

print(arr1.sort()) # 就地进行排序，修改原数组)
print("")
print("")
print(arr1)
print("++++++++++++++++++++++++++++++++++")
"""
多维数组排序

"""
arr2 = np.random.randn(3, 4)
print(arr2)
print("")
print(np.sort(arr2, axis=1)) # 按列排序)
print("")
print(np.sort(arr2, axis=0)) # 按行排序)


my_dt = np.dtype([('name',  'U10'),('age',  int)]) # 按关键字排序
print(my_dt)
print("")
print("")
"""
按键排序
"""
arr3 = np.array([("罗志祥", 21), ("黄渤", 25), ("孙红雷",  17),  ("黄磊",27)], dtype=my_dt)
print(arr3)
print("")
print("")
print(np.sort(arr3, order='age'))


















































































