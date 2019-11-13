# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/13 16:43'


import numpy as  np

"""
#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)

使用tile函数创建一个8*8的棋盘式矩阵
"""

# tile() 知识点
z_1 = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(z_1)
print("************************")



"""

#### 22. Normalize a 5x5 random matrix (★☆☆)

归一化 一5*5 的随机矩阵

"""


z_2 = np.random.random((5,5))
z_2 = (z_2 - np.mean (z_2)) / (np.std (z_2))
print(z_2)
print("************************")

"""
23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)
创建一个自定义dtype，将颜色描述为四个无符号字节(RGBA)




"""

# dtype 知识点
color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])


print(color)

print("************************")

"""
#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)
5x3矩阵乘以3x2矩阵(真实矩阵产品)

"""

# dot知识点
z_3 = np.dot(np.ones((5,3)), np.ones((3,2)))
print(z_3)
print("************************")

# Alternative solution, in Python 3.5 and above
z_4 = np.ones((5,3)) @ np.ones((3,2))
print(z_4)
print("************************")


"""
25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)

给定一个一维数组
把所有的3到8之间的元素取负数
"""

z_5 = np.arange(11)
print(z_5)
z_5[(3 < z_5) & (z_5 <= 8)] *= -1
print(z_5)
print("************************")

"""
26. What is the output of the following script? (★☆☆)
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
"""

print(sum(range(5),-1))

# 细节问题
from numpy import *
print(sum(range(5),-1))
print("************************")

"""
27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)


Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
"""

"""
28. What are the result of the following expressions?

np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)


"""


print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))
print("************************")


"""
#### 29. How to round away from zero a float array ? (★☆☆)

浮点数取整数

"""

z_6 = np.random.uniform(-10,+10,10)
print(z_6)
print (np.copysign(np.ceil(np.abs(z_6)), z_6))
print("************************")


"""
30. How to find common values between two arrays? (★☆☆)
找出两个数组之中的相同元素
"""
print('         30          ')

# 有问题

z_7 = np.random.randint(0,10,10)
z_8 = np.random.randint(0,10,10)
print(z_7)
print(z_8)
print(np.intersect1d(z_7,z_8))

print("************************")

