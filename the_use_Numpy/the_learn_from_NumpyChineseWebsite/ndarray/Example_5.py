# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/15 14:25'



import numpy as np


"""
布尔数组索引
当obj是Boolean类型的数组对象(例如可以从比较运算符返回)时， 会发生这种高级索引。 
单个布尔索引数组实际上与 x[obj.nonzero()] 相同， 其中，如上所述，obj.nonzero() 
返回整数索引数组的元组(长度为 obj.ndim)， 显示obj的 True 元素。但是，当 obj.shape == x.shape 时，它会更快。

如果 obj.ndim == x.ndim，x[obj] 返回一个1维数组， 其中填充了与obj的 True 值对应的x的元素。 
搜索顺序为 row-major， C样式。如果 obj 在 x 的界限之外的条目上有True值， 则会引发索引错误。
如果 obj 小于 x ，则等同于用False填充它。


"""

"""
一个常见的用例是过滤所需的元素值。例如，可能希望从数组中选择非NaN的所有条目：

"""

data = np.array([[1., 2.], [np.nan, 3.], [np.nan, np.nan]])

# print(data[~np.isnan(data)])

"""
输出
[1. 2. 3.]
"""

# print(np.isnan(data))

"""
输出
[[False False]
 [ True False]
 [ True  True]]
"""



"""
或者希望为所有负面元素添加常量：

"""

data1 = np.array([1., -1., -2., 3])
# print(data1 < 0)
"""
输出

[False  True  True False]

"""

data1[data1 < 0] += 20

# print(data1)
"""
输出
[ 1. 19. 18.  3.]
"""


"""
通常，如果索引包含布尔数组，则结果将与将 obj.nonzero() 插入到相同位置并使用上述整数组索引机制相同。 
x[ind_1，boolean_array，ind_2] 等价于 x[(ind_1，)+boolean_array.nonzero()+(ind_2，)]。

如果只有一个布尔数组且没有整数索引数组，则这是直截了当的。必须注意确保布尔索引具有与其应该使用的维度 完全相同的 维度。


"""

"""
从数组中，选择总和小于或等于2的所有行：

"""

data2 = np.array([[0, 1], [1, 1], [2, 2]])
# np.sum
rowsum = data2.sum(-1)
# print(rowsum)
"""
输出
[1 2 4]

每组行相加的结果
"""

# print(rowsum <= 2)

"""
输出

[ True  True False]
"""
# print(data2[rowsum <= 2, :])
"""
输出
[[0 1]
 [1 1]]
"""



"""
使用布尔索引选择加起来为偶数的所有行。同时，应使用高级整数索引选择列0和2。使用该ix_功能可以通过以下方式完成：
"""

data3 = np.array([[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8],
            [ 9, 10, 11]])
rows = (data3.sum(-1) % 2) == 0
print(rows)
"""
输出
[False  True False  True]
"""
# 限定 0 2 列
columns = [0, 2]
print(data3[np.ix_(rows, columns)])

"""
输出
[[ 3  5]
 [ 9 11]]
"""


