# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/15 13:53'

import numpy as np


"""
结合高级索引和基本索引
当至少有一个slice（:），省略号（...）或newaxis 索引
（或者数组的维度多于高级索引）时，行为可能会更复杂。这就像连接每个高级索引元素的索引结果一样

在最简单的情况下，只有一个 单一的 指标先进。单个高级索引可以例如替换切片，
并且结果数组将是相同的，但是，它是副本并且可以具有不同的存储器布局。当可能时，切片是优选的。
"""


data = np.arange(16).reshape((4,4))

# print(data)

"""
输出
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
 
"""

# print(data[1:2,1:3])

"""
输出
[[5 6]]
"""

# print(data[1:2, [1, 2]])

"""
输出
[[5 6]]
"""

print(data[2:4,2:4])
"""
输出

[[10 11]
 [14 15]]
 
 2:4   后面不取
 第 2  3 行
 
 第2 3 列
 
 

"""

"""
了解情况的最简单方法可能是考虑结果形状。索引操作分为两部分，
即由基本索引（不包括整数）定义的子空间和来自高级索引部分的子空间。需要区分两种索引组合：

高级索引由切片分隔，Ellipsis或newaxis。例如。x[arr1, :, arr2]
高级索引都是相邻的。例如 x[..., arr1, arr2, :]，但不是 x[arr1, :, 1]，因为 1 是这方面的高级索引。
在第一种情况下，高级索引操作产生的维度首先出现在结果数组中，然后是子空间维度。 
在第二种情况下，高级索引操作的维度将插入到结果数组中与初始数组中相同的位置
（后一种逻辑使简单的高级索引行为就像切片一样）。
"""


"""
示例：

假设 x.shape 是(10，20，30)， 并且 ind 是（2,3,4）形状的索引 intp 数组， 
那么 result = x[..., ind, : ] 具有形状(10，2，3，4，30)， 因为(20，)形状子空间已经被(2，3，4)形状的广播索引子空间所取代。 
如果我们让i，j，k在(2，3，4)形状子空间上循环，则结果 result[...,i,j,k,:] = x[...,ind[i,j,k],:] 。 
本例产生的结果与 x.take(ind，axis=-2) 相同。

设x.shape（10,20,30,40,50）并假设ind_1 并ind_2可以广播到形状（2,3,4）。然后 x[:,ind_1,ind_2]具有形状（10,2,3,4,40,50），
因为来自X的（20,30）形子空间已经被索引的（2,3,4）子空间替换。但是，它 x[:,ind_1,:,ind_2]具有形状（2,3,4,10,30,50），
因为在索引子空间中没有明确的位置，所以它在开头就被添加了。始终可以使用 .transpose()在任何需要的位置移动子空间。 
请注意，此示例无法使用复制take。
"""













































































































