# -*- coding: utf-8 -*-
# @Time  : 2019/11/12 19:56
# @Author : Mr.Lin


"""

数学和统计方法

"""

import numpy as np

"""
生成一些随机正态分布的数据，然后做聚类统计
"""

arr = np.random.rand(5,4)

print(arr)
print("======================")

print(arr.mean())
print("======================")

print(np.mean(arr))
print("======================")

print(arr.sum())
print("======================")

"""
mean以及sum函数可以接受一个axis选项参数
用于计算该轴向上的统计值
最终结果少是一个少一维的数组

"""

print(arr.mean(axis=1))
print("======================")


print(arr.sum(axis=0))
print("======================")

"""
arr.mean(1)是计算行的平均值
arr.sum(0)是计算列的和

"""













