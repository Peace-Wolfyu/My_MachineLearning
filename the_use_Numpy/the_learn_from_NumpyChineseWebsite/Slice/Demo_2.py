# -*- coding: utf-8 -*-
# @Time  : 2019/11/15 20:39
# @Author : Mr.Lin

import numpy as np


"""
整型数组访问：当我们使用切片语法访问数组时，
得到的总是原数组的一个子集。
整型数组访问允许我们利用其它数组的数据构建一个新的数组
"""

a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[
          [0, 1, 2], [0, 1, 0]
      ])  # Prints "[1 4 5]"
# >>>
# [1 4 5]


print(" ")


# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"
# >>>
# [1 4 5]

print(" ")
# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"

"""
整型数组访问语法还有个有用的技巧，可以用来选择或者更改矩阵中每行中的一个元素：

"""
# Create a new array from which we will select elements
a_1 = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(a_1)  # prints "array([[ 1,  2,  3],
         #                [ 4,  5,  6],
         #                [ 7,  8,  9],
         #                [10, 11, 12]])"

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a_1[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a_1[np.arange(4), b] += 10

print(a_1)  # prints "array([[11,  2,  3],
         #                [ 4,  5, 16],
         #                [17,  8,  9],
         #                [10, 21, 12]])






































