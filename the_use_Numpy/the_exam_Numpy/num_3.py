# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/13 16:43'


import numpy as  np

"""
#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)

使用tile函数创建一个8*8的棋盘式矩阵
"""

# tile() 知识点

"""

        def tile(A, reps):
            通过重复 A 值 (reps次数) 来构造一个数组。
            
            如果reps的长度为d，则结果的尺寸为max（d，A.ndim）。
            如果``A.ndim <d''，则通过添加新轴将A提升为d维。
            因此，将形状（3，）阵列提升为（1，3）以进行2D复制，或将形状（1、1、3）提升为3D复制。
            如果这不是理想的行为，则在调用此函数之前，将A手动提升为d维度。
            如果``A.ndim> d''，则通过在其前面加上1来将reps提升为A.ndim。
            因此，对于形状为（2、3、4、5）的A，将（2、2）的reps视为（1、1、2、2）。
            
            注意：尽管可以将tile用于广播，但强烈建议使用numpy的广播操作和功能。
            
            Parameters
    ----------
    A : array_like
        The input array.
    reps : array_like
        The number of repetitions of `A` along each axis.

    Returns
    -------
    c : ndarray
        The tiled output array.
        
        
     Examples
    --------
    >>> a = np.array([0, 1, 2])
    >>> np.tile(a, 2)
    array([0, 1, 2, 0, 1, 2])
    >>> np.tile(a, (2, 2))
    array([[0, 1, 2, 0, 1, 2],
           [0, 1, 2, 0, 1, 2]])
    >>> np.tile(a, (2, 1, 2))
    array([[[0, 1, 2, 0, 1, 2]],
           [[0, 1, 2, 0, 1, 2]]])

    >>> b = np.array([[1, 2], [3, 4]])
    >>> np.tile(b, 2)
    array([[1, 2, 1, 2],
           [3, 4, 3, 4]])
    >>> np.tile(b, (2, 1))
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]])

    >>> c = np.array([1,2,3,4])
    >>> np.tile(c,(4,1))
    array([[1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4]])


"""
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

"""

        def dot(a, b, out=None):
        计算两个数组的点积
        
        如果“ a”和“ b”都是一维数组，则它是向量的内积（无复共轭）。
        如果a和b都是二维数组，则是矩阵乘法，但是最好使用：func：matmul或a @ b。
        如果`a`或`b`均为0-D（标量），则等效于：func：`multiply`并使用``numpy.multiply（a，b）``或``a * b``为首选。
        如果“ a”是一个N维数组，而“ b”是一维数组，则它是“ a”和“ b”最后一条轴上的总和。
        如果a是ND数组，b是MD数组（其中M> = 2），则它是a的最后一个轴和a的倒数第二个轴的和积。`b` ::
                dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
        
        
        Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.
    out : ndarray, optional
    
    输出参数。如果没有使用，它必须具有返回的确切类型。
    特别是，它必须具有正确的类型，必须是C连续的，
    并且它的dtype必须是为dot（a，b）返回的dtype。这是一项性能功能。因此，
    如果不满足这些条件，则会引发异常，而不是尝试变得灵活。
    
    
    
     Examples
    --------
    >>> np.dot(3, 4)
    12

    Neither argument is complex-conjugated:

    >>> np.dot([2j, 3j], [2j, 3j])
    (-13+0j)

    For 2-D arrays it is the matrix product:

    >>> a = [[1, 0], [0, 1]]
    >>> b = [[4, 1], [2, 2]]
    >>> np.dot(a, b)
    array([[4, 1],
           [2, 2]])

    >>> a = np.arange(3*4*5*6).reshape((3,4,5,6))
    >>> b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
    >>> np.dot(a, b)[2,3,2,1,2,2]
    499128
    >>> sum(a[2,3,2,:] * b[1,2,:,2])
    499128
    
    
    


"""
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
"""
上下两个sum不同

"""
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

"""
        
        def uniform(low=0.0, high=1.0, size=None): # real signature unknown; restored from __doc__
                刻画一些服从均匀分布的数据
                
                
            样本均匀分布在半开区间``[low，high）''（包括低，但不包括高）。换句话说，给定间隔内的任何值都可能由“统一”得出。
            
            参数：
            low : float or array_like of floats, optional
            输出间隔的下边界。生成的所有值都将大于或等于低。默认值为0。
            
            high : float or array_like of floats
            输出间隔的上限。生成的所有值将小于高。默认值为1.0。
            
            size : int or tuple of ints, optional
            输出形状。
            如果给定的形状是例如（m，n，k），则绘制m * n * k个样本。
            如果size为``None''（默认），则如果``low''和``high''均为标量，则返回单个值。
            否则，将绘制“ np.broadcast（low，high）.size”样本。

 
    
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



"""
            def randint(low, high=None, size=None, dtype='l'): # real signature unknown; restored from __doc__
            从“低”（包含）到“高”（不含）返回随机整数。
            
            从指定的dtype的“离散均匀”分布中的“半开”间隔[[低]，“高”）
            返回随机整数。如果`high`为None（默认值），则结果来自[0，`low`）。
            Parameters
            ----------
            low : int or array-like of ints
            从分布中得出的最低（带符号）整数（除非“ high = None”，在这种情况下，此参数比“最高”这样的整数高一个）。
            
            high : int or array-like of ints, optional
            如果提供，则从分布中得出的最大（有符号）整数之上的一个（如果``high = None''，请参见上面的行为）。
            如果为数组，则必须包含整数值 

            size : int or tuple of ints, optional
            输出形状。如果给定的形状是例如（m，n，k），则绘制m * n * k个样本。默认值为无，在这种情况下，将返回单个值。
            
            dtype : dtype, optional
            
             Examples
            --------
            >>> np.random.randint(2, size=10)
            array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0]) # random
            >>> np.random.randint(1, size=10)
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
            Generate a 2 x 4 array of ints between 0 and 4, inclusive:
    
            >>> np.random.randint(5, size=(2, 4))
            array([[4, 0, 2, 1], # random
                   [3, 2, 2, 0]])
    
            Generate a 1 x 3 array with 3 different upper bounds
    
            >>> np.random.randint(1, [3, 5, 10])
            array([2, 2, 9]) # random
    
            Generate a 1 by 3 array with 3 different lower bounds
    
            >>> np.random.randint([1, 5, 7], 10)
            array([9, 8, 7]) # random
    
            Generate a 2 by 4 array using broadcasting with dtype of uint8
    
            >>> np.random.randint([1, 3, 5, 7], [[10], [20]], dtype=np.uint8)
            array([[ 8,  6,  9,  7], # random
                   [ 1, 16,  9, 12]], dtype=uint8)
                   
                   
                   


"""
z_7 = np.random.randint(0,10,10)
z_8 = np.random.randint(0,10,10)
print(z_7)
print(z_8)



"""

        def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
                找到两个数组的交集。
                
                Parameters
    ----------
            ar1, ar2 : array_like
                输入数组。如果尚未为1D，则将被展平。
                
            assume_unique : bool
                如果为True，则假定输入数组都是唯一的，这可以加快计算速度。默认值为False。
                
            return_indices : bool
                如果为True，则返回与两个数组的交集相对应的索引。如果有多个值，则使用值的第一个实例。默认值为False。
        

    Examples
    --------
    >>> np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
    array([1, 3])

    To intersect more than two arrays, use functools.reduce:

    >>> from functools import reduce
    >>> reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
    array([3])

    To return the indices of the values common to the input arrays
    along with the intersected values:

    >>> x = np.array([1, 1, 2, 3, 4])
    >>> y = np.array([2, 1, 4, 6])
    >>> xy, x_ind, y_ind = np.intersect1d(x, y, return_indices=True)
    >>> x_ind, y_ind
    (array([0, 2, 4]), array([1, 0, 2]))
    >>> xy, x[x_ind], y[y_ind]
    (array([1, 2, 4]), array([1, 2, 4]), array([1, 2, 4]))
 

"""
print(np.intersect1d(z_7,z_8))

print("************************")

