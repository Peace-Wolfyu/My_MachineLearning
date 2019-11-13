# -*- coding: utf-8 -*-
# @Time  : 2019/11/13 19:33
# @Author : Mr.Lin


import numpy as np


"""
#### 41. How to sum a small array faster than np.sum? (★★☆)

对一个小数据量的数组进行求和操作
要比numpy的sum函数运算快
"""
Z1 = np.arange(10)
# reduce知识点
sum1 = np.add.reduce(Z1)
print(Z1)
print(sum1)
print("********************************************************")


"""
#### 42. Consider two random array A and B, check if they are equal (★★☆)

给定两个随机数组 A B
检查它们是否相等
"""

A = np.random.randint(0,2,5)
print(A)
B = np.random.randint(0,2,5)
print(B)
# Assuming identical shape of the arrays and a tolerance for the comparison of values
equal = np.allclose(A,B)
print(equal)

# Checking both the shape and the element values, no tolerance (values have to be exactly equal)
equal = np.array_equal(A,B)
print(equal)

print("********************************************************")

"""
#### 43. Make an array immutable (read-only) (★★☆)

使得一个数组不可变
"""
Z2 = np.zeros(10)
print(Z2)
Z2.flags.writeable = False
# 此时进行修改操作则会报错
# Z2[0] = 1
print("********************************************************")

"""
#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)
给定一个 10 * 2 的矩阵 
把它从直角坐标的形式转换成极坐标的形式
"""

Z3 = np.random.random((10,2))
print(Z3)
X,Y = Z3[:,0], Z3[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)
print("********************************************************")

"""
#### 45. Create a random vector of size 10 and replace the maximum value by 0 (★★☆)

创建一个大小为10的随机数组向量
把最大值用0替换
"""
Z4 = np.random.random(10)
print(Z4)
Z4[Z4.argmax()] = 0
print(Z4)
print("********************************************************")

"""
#### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆)
创建一个包含“x”和“y”坐标的结构化数组，覆盖\[0,1\]x\[0,1\]区域



"""
Z5 = np.zeros((5,5), [('x',float),('y',float)])
print(Z5)
print("--------------------")
Z5['x'], Z5['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(Z5)

print("********************************************************")

"""
####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))
给定两个数组X和Y，构造柯西矩阵C (Cij =1/(xi - yj))
"""

X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))
print("********************************************************")

"""
#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)
打印每个numpy标量类型的最小和最大可表示值
"""
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
print("------------------")
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)
print("********************************************************")

"""
#### 49. How to print all the values of an array? (★★☆)

如何打印数组中所有的值
"""

np.set_printoptions(threshold=np.nan)
Z6 = np.zeros((16,16))
print(Z6)
print("********************************************************")

"""
#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)
在一个向量中找到最接近给定标量的值
"""
Z7 = np.arange(100)
print(Z7)
print("------------------")
v = np.random.uniform(0,100)
index = (np.abs(Z7-v)).argmin()
print(Z7[index])




















































