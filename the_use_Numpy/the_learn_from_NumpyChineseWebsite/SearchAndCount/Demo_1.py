# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 15:54'


"""
搜索

argmax(a[, axis, out])

返回沿轴的最大值的索引

nanargmax(a[, axis])

返回指定轴上最大值的索引，忽略NAN

argmin(a[, axis, out])

返回沿轴的最小值的索引

nanargmin(a[, axis])

返回指定轴上的最小值的索引，忽略NAN

argwhere(a)

返回按元素分组的非零数组元素的索引。

nonzero(a)

返回非零元素的索引

flatnonzero(a)

返回a的展平版本中非零的索引

where(condition, [x, y])

返回元素，可以是x或y，具体取决于条件

searchsorted(a, v[, side, sorter])

查找应插入元素以维护顺序的索引

extract(condition, arr)

返回满足某些条件的数组元素

计数

count_nonzero(a[, axis])
计算数组a中的非零值的数量
"""

import numpy as np

arr = np.arange(6).reshape(2,3) + 10
# print(arr)
# >>>
# [[10 11 12]
#  [13 14 15]]

# print(np.argmax(arr)) # axis默认为None，返回flatten之后的索引)
# >>> 5

# print(np.argmax(arr, axis=0))
# >>> [1 1 1]

# print(np.argmax(arr, axis=1))
# >>> [2 2]

# print(# argmin类似于argmax
# np.argmin(arr))
#
# >>> 0

# print(np.argmin(arr, axis=1))
# >>>  [0 0]


arr2 = np.array([[10, np.NaN, 11, 12],
                 [13, 14, 15, np.NaN]])


# print(arr2)
# >>>
# [[10. nan 11. 12.]
#  [13. 14. 15. nan]]

# print(np.argmax(arr2, axis=0) # 不会忽略NaN值
# )
# >>> [1 0 1 1]

# print(np.nanargmax(arr2, axis=0)
# )
# >>> [1 1 1 0]

# print(np.argwhere(arr > 12) # 返回的是元素的坐标
# )
#
# >>> [[1 0]
#  [1 1]
#  [1 2]]

# print(np.nonzero(arr) # 返回数组的元组
# )
#
# >>>
# (array([0, 0, 0, 1, 1, 1], dtype=int32), array([0, 1, 2, 0, 1, 2], dtype=int32))


# print(np.flatnonzero(arr) # 等效于 np.nonzero(np.ravel(arr))[0]
# )
# >>>
# [0 1 2 3 4 5]

arr3 = np.array([1, 2, 3, 4, 5])
arr4 = np.array(['a', 'b', 'c', 'd', 'e'])
cond = np.array([True, False, True, False, True])

result = np.where(cond, arr3, arr4)

# print(result)
#
# >>>
# ['1' 'b' '3' 'd' '5']

# print(# 将arr1中大于3的全部变为-3
# np.where(arr3 > 3, -3, arr3))
# >>>
# [ 1  2  3 -3 -3]


# print(# 将arr1中大于3的全部变为该值的相反数
# np.where(arr3 > 3, -arr3, arr3))
#
# >>>
# [ 1  2  3 -4 -5]


# print(np.searchsorted([1,2,3,4,5], 3)
# )
# >>>
# 2

# print(np.searchsorted([1,2,3,4,5], 3, side='right')
# )
# >>>
# 3

# print(np.searchsorted([1,2,3,4,5], [-10, 10, 2, 3])
# )
# >>>
# [0 5 1 2]


# print(np.extract(arr > 12, arr)
# )
# >>>
# [13 14 15]

# print(np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]]) # 默认axis为None，统计flatten之后的数组
# )
#
# >>>
# 5

# print(np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]], axis=0)
# )
#
# >>>
# [1 1 1 1 1]


# print(np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]], axis=1)
# )
#
# >>>
# [2 3]






































