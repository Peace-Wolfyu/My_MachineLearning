# -*- coding: utf-8 -*-
# @Time  : 2019/11/13 20:10
# @Author : Mr.Lin


import numpy as np

"""
51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)

创建一个表示位置(x,y)和颜色(r,g,b)的结构化数组

"""

Z1 = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z1)

print("**********************************************************************************")


"""
52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)

考虑一个结构为 100 * 2 的随机向量，根据点查找距离

"""

Z2 = np.random.random((10,2))
print(Z2)
print("-----------------------------------")
X,Y = np.atleast_2d(Z2[:,0], Z2[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)
print("-----------------------------------")

# Much faster with scipy
import scipy
# Thanks Gavin Heverly-Coulson (#issue 1)
import scipy.spatial

Z3 = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z3,Z3)
print(D)

print("**********************************************************************************")


"""
#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?

如何原地把一个32位浮点型的数组转换成32位整数型
"""
Z4 = np.arange(10, dtype=np.float32)
print(Z4)
print("-----------------------------------")

Z4 = Z4.astype(np.int32, copy=False)
print(Z4)

print("**********************************************************************************")

"""
#### 54. How to read the following file? (★★☆)


```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```
"""
from io import StringIO

# Fake file
s = StringIO("""1, 2, 3, 4, 5\n
                6,  ,  , 7, 8\n
                 ,  , 9,10,11\n""")
Z5 = np.genfromtxt(s, delimiter=",", dtype=np.int)
print(Z5)

print("**********************************************************************************")

"""
#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)

和numpy数组等效的枚举值

"""
Z6= np.arange(9).reshape(3,3)
print(Z6)
print("-----------------------------------")
for index, value in np.ndenumerate(Z6):
    print(index, value)
print("-----------------------------------")
for index in np.ndindex(Z6.shape):
    print(index, Z6[index])
print("**********************************************************************************")


"""
#### 56. Generate a generic 2D Gaussian-like array (★★☆)


"""

X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)
print("**********************************************************************************")


"""
#### 57. How to randomly place p elements in a 2D array? (★★☆)

如何在2维数组里随机的替换一个值
"""

n = 10
p = 3
Z7 = np.zeros((n,n))
print(Z7)
print("-----------------------------------")
np.put(Z7, np.random.choice(range(n*n), p, replace=False),1)
print(Z7)
print("**********************************************************************************")

"""
#### 58. Subtract the mean of each row of a matrix (★★☆)

计算一个矩阵每行的平均值

"""
# Author: Warren Weckesser

X = np.random.rand(5, 10)
print(X)
# Recent versions of numpy
Y = X - X.mean(axis=1, keepdims=True)

# Older versions of numpy
Y = X - X.mean(axis=1).reshape(-1, 1)

print("**********************************************************************************")

"""
#### 59. How to sort an array by the nth column? (★★☆)


"""

Z8 = np.random.randint(0,10,(3,3))
print(Z8)
print("-----------------------------------")
print(Z8[Z8[:,1].argsort()])
print("**********************************************************************************")

"""
#### 60. How to tell if a given 2D array has null columns? (★★☆)

如何确认一个给定的2维数组是否有空的列
"""



Z9 = np.random.randint(0,3,(3,10))
print(Z9)
print("-----------------------------------")
print((~Z9.any(axis=0)).any())


































































