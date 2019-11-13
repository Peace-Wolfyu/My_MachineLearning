# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/13 16:20'

import numpy as np

"""
#### 11. Create a 3x3 identity matrix (★☆☆)

创建一个 3 * 3 的单位矩阵
"""


# eye() 知识点
z = np.eye(3)
print(z)

print("************************")



"""
#### 12. Create a 3x3x3 array with random values (★☆☆)

创建一个包含随机数值 的 3*3*3 数组

"""

# random() 知识点
z_1 = np.random.random((3,3,3))
print(z_1)
print("************************")


"""
13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)

创建一个 10 * 10 的数组，里面是随机数组
找出最大以及最小值
"""

z_2 = np.random.random((10,10))
print(z_2)
# max() min() 知识点
zMax,zMin = z_2.max(),z_2.min()
print(zMax)
print(zMin)
print("************************")


"""
14. Create a random vector of size 30 and find the mean value (★☆☆)
创建一个大小为30的随机数值向量，找出平均值
"""

z_3 = np.random.random(30)
# mean() 知识点
print(z_3.mean())
print("************************")


"""
15. Create a 2d array with 1 on the border and 0 inside (★☆☆)

创建一个二维数组  1 在 边界  0 在里面
"""


z_4 = np.ones((10,10))
print(z_4)

# 切片[1:-1,1:-1]知识点
z_4[1:-1,1:-1] = 0
print(z_4)
print("************************")


"""
16. How to add a border (filled with 0's) around an existing array? (★☆☆)

给已经存在的一个数组添加 边界 全是 0
"""

z_5 = np.ones((3,3))
print(z_5)

# pad() 知识点
z_5 = np.pad(z_5, pad_width=1, mode='constant', constant_values=0)
print(z_5)
print("************************")

"""
17. What is the result of the following expression? (★☆☆)


一下表达式的结果是？

0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1


"""


print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)

print("************************")

"""
18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)

创建 5*5的矩阵 
对角线下面的数值设置为 1 2 3 4
"""

# diag() 知识点
z_6 = np.diag(1+np.arange(4),k=-1)
print(z_6)
print("************************")


"""
19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)

创建一个 8*8的矩阵 使得呈现出棋盘的形式
"""


z_7 = np.zeros((8,8),dtype=int)
z_7[1::2,::2] = 1
z_7[::2,1::2] = 1
print(z_7)
print("************************")


"""
20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?

对于一个 6*7*8 的数组
第100个元素的索引(x,y,z)是什么
"""


# unravel() 知识点
print(np.unravel_index(99,(6,7,8)))



print("************************")

