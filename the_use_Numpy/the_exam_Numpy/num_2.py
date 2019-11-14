# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/13 16:20'

import numpy as np

"""
#### 11. Create a 3x3 identity matrix (★☆☆)

创建一个 3 * 3 的单位矩阵

identity matrix   ---> 单位矩阵
"""


# eye() 知识点

"""
        def eye(N, M=None, k=0, dtype=float, order='C'):
            返回一个二维数组，对角线上有值为 1 ，其他地方为零。
        参数：
            N : int
            输出中的行数。
            
            M : int, optional
            输出中的列数。如果为None，则默认为`N`。
            
            k : int, optional
            对角线的索引：0（默认值）表示主对角线，正值表示上对角线，负值表示下对角线。
            
            dtype : data-type, optional
            order : {'C', 'F'}, optional

            Examples
    --------
    >>> np.eye(2, dtype=int)
    array([[1, 0],
           [0, 1]])
    >>> np.eye(3, k=1)
    array([[0.,  1.,  0.],
           [0.,  0.,  1.],
           [0.,  0.,  0.]])
"""
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

"""
        def ones(shape, dtype=None, order='C'):
        返回给定形状和类型的新数组，并填充为1。
        
        参数：
          shape：  int or sequence of ints
          dtype : data-type, optional   Default is `numpy.float64`.
          order : {'C', 'F'}, optional, default: C
        
        See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    empty : Return a new uninitialized array.
    zeros : Return a new array setting values to zero.
    full : Return a new array of given shape filled with value.
    
     Examples
    --------
    >>> np.ones(5)
    array([1., 1., 1., 1., 1.])

    >>> np.ones((5,), dtype=int)
    array([1, 1, 1, 1, 1])

    >>> np.ones((2, 1))
    array([[1.],
           [1.]])

    >>> s = (2,2)
    >>> np.ones(s)
    array([[1.,  1.],
           [1.,  1.]])
           
           

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

"""
ndarray = numpy.pad(array, pad_width, mode, **kwargs)

    array为要填补的数组

        pad_width是在各维度的各个方向上想要填补的长度,如（（1，2），（2，2）），
        表示在第一个维度上水平方向上padding=1,垂直方向上padding=2,
        在第二个维度上水平方向上padding=2,垂直方向上padding=2。如果直接输入一个整数，
        则说明各个维度和各个方向所填补的长度都一样。
        mode为填补类型，即怎样去填补，有“constant”，“edge”等模式，如果为constant模式，
        就得指定填补的值，如果不指定，则默认填充0。 

剩下的都是一些可选参数，具体可查看 
https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html

ndarray为填充好的返回值。

"""
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

"""
        def diag(v, k=0):提取对角线或构造对角线数组。
        
        参数：
            v : array_like
            如果v是二维数组，则返回其第k个对角线的副本。
            如果v是一维数组，则返回第k个带有v的二维数组。
            
            k : int, optional
            对角线问题。
            默认值为0。对于主对角线上方的对角线，请使用“ k> 0”，对于主对角线下方的对角线，请使用“ k <0”。
            
             Examples
    --------
    >>> x = np.arange(9).reshape((3,3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> np.diag(x)
    array([0, 4, 8])
    >>> np.diag(x, k=1)
    array([1, 5])
    >>> np.diag(x, k=-1)
    array([3, 7])

    >>> np.diag(np.diag(x))
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])
           
           

"""
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

"""
            unravel_index(indices, shape, order='C')
                将平面索引或平面索引数组转换为坐标数组的元组。
            参数：
                indices : array_like
                    一个整数数组，其元素是尺寸为``shape''的数组的展平版本的索引。在1.6.0版之前，此函数仅接受一个索引值。
                shape : tuple of ints
                用于解散``索引''的数组的形状。
                order : {'C', 'F'}, optional

 Examples
    --------
    >>> np.unravel_index([22, 41, 37], (7,6))
    (array([3, 6, 6]), array([4, 5, 1]))
    >>> np.unravel_index([31, 41, 13], (7,6), order='F')
    (array([3, 6, 6]), array([4, 5, 1]))

    >>> np.unravel_index(1621, (6,7,8,9))
    (3, 1, 4, 1)

            
"""
print(np.unravel_index(99,(6,7,8)))



print("************************")

