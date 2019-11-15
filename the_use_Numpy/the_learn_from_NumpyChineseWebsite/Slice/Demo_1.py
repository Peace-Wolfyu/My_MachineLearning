# -*- coding: utf-8 -*-
# @Time  : 2019/11/15 20:28
# @Author : Mr.Lin

import numpy as np


"""
访问数组
Numpy提供了多种访问数组的方法。

切片：和Python列表类似，numpy数组可以使用切片语法。因为数组可以是多维的，所以你必须为每个维度指定好切片。
"""

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array(
    [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12]
    ]
)

# print(a)
# >>>
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
print("")
print("")
print("")

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
# 选取前两行，第1 2 列
b = a[:2, 1:3]
# print(b)
print("")
print("")
print("")
# >>>
# [[2 3]
#  [6 7]]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.

# 切片操作会影响原数据

# print(a[0, 1])   # Prints "2"
b[0, 0] = 77    # b[0, 0] is the same piece of data as a[0, 1]
# print(a[0, 1])   # Prints "77"

"""
你可以同时使用整型和切片语法来访问数组。但是，
这样做会产生一个比原数组低阶的新数组。需要注意的是，这里和MATLAB中的情况是不同的：
"""

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a_1 = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# >>>
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:

# 选取第一行的元素
row_r1 = a_1[1, :]    # Rank 1 view of the second row of a
# 同上
row_r2 = a_1[1:2, :]  # Rank 2 view of the second row of a
# print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"

# >>>
# [5 6 7 8] (4,)

print(" ")
print(" ")
print(" ")

# print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
# 选取第一列的值
col_r1 = a_1[:, 1]
col_r2 = a_1[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # Prints "[[ 2]
                            #          [ 6]
                            #          [10]] (3, 1)"










