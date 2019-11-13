# -*- coding: utf-8 -*-
# @Time  : 2019/11/13 20:29
# @Author : Mr.Lin

import numpy as np


"""
#### 61. Find the nearest value from a given value in an array (★★☆)

从数组中找出最接近给定数的一个数
"""

Z1 = np.random.uniform(0,1,10)
z = 0.5
m = Z1.flat[np.abs(Z1 - z).argmin()]
print(Z1)
print("-----------------------------------")
print(m)
print("**********************************************************************************")


"""
#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)
给定两个数组
一个 1*3 一个 3*1
如何通过迭代器计算它们两个的和
"""
A = np.arange(3).reshape(3,1)
print(A)
print("-----------------------------------")
B = np.arange(3).reshape(1,3)
print(B)
print("-----------------------------------")
it = np.nditer([A,B,None])
for x,y,z in it: z[...] = x + y
print(it.operands[2])
print("**********************************************************************************")

"""
#### 63. Create an array class that has a name attribute (★★☆)

创建一个由名字属性的数组类
"""
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")


Z2 = NamedArray(np.arange(10), "range_10")
print (Z2.name)
print("**********************************************************************************")

"""
#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector
 (be careful with repeated indices)? (★★★)

给定一个向量
如何通过索引的形式给每个元素 +1

"""
# Author: Brett Olsen

Z3 = np.ones(10)
print(Z3)
print("-----------------------------------")
I = np.random.randint(0,len(Z3),20)
Z3 += np.bincount(I, minlength=len(Z3))
print(Z3)
print("-----------------------------------")

# Another solution
# Author: Bartosz Telenczuk
np.add.at(Z3, I, 1)
print(Z3)

print("**********************************************************************************")

"""
#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)

通过索引列表I来计算向量X和数组F
"""


X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print(F)

print("**********************************************************************************")

"""
#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★)
考虑(dtype=ubyte)的一个(w,h,3)图像，计算唯一颜色的数量

"""

w,h = 16,16
I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
n = len(np.unique(F))
print(np.unique(I))
print("**********************************************************************************")


"""
#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)


"""
A = np.random.randint(0,10,(3,4,3,4))
print(A)
print("-----------------------------------")
# solution by passing a tuple of axes (introduced in numpy 1.7.0)
sum = A.sum(axis=(-2,-1))
print(sum)
print("-----------------------------------")
# solution by flattening the last two dimensions into one
# (useful for functions that don't accept tuples for axis argument)
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)
print("**********************************************************************************")


"""
#### 68. Considering a one-dimensional vector D, how to compute means of subsets of 
D using a vector S of same size describing subset  indices? (★★★)

考虑一维向量D，如何计算其子集的均值

使用相同大小的向量S来描述子集
"""
D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)
print("-----------------------------------")

# Pandas solution as a reference due to more intuitive code
import pandas as pd
print(pd.Series(D).groupby(S).mean())

print("**********************************************************************************")


"""
#### 69. How to get the diagonal of a dot product? (★★★)
如何得到点积的对角线

"""

A = np.random.uniform(0,1,(5,5))
B = np.random.uniform(0,1,(5,5))

# Slow version
np.diag(np.dot(A, B))

# Fast version
np.sum(A * B.T, axis=1)

# Faster version
np.einsum("ij,ji->i", A, B)


"""
#### 70. Consider the vector 
\[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)

[1, 2, 3, 4, 5\]，如何建立一个新的向量与3个连续的零交错之间的每个值


"""
Z4 = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z4) + (len(Z4)-1)*(nz))
Z0[::nz+1] = Z4
print(Z0)






























































































