# -*- coding: utf-8 -*-
# @Time  : 2019/11/13 19:14
# @Author : Mr.Lin

import numpy as np


"""
31. How to ignore all numpy warnings (not recommended)? (★☆☆)
怎样忽略numpy的警告（这是不推荐的做法）
"""

# Suicide mode on
# defaults = np.seterr(all="ignore")
# Z = np.ones(1) / 0

# Back to sanity
# _ = np.seterr(**defaults)



"""
32. Is the following expressions true? (★☆☆)

np.sqrt(-1) == np.emath.sqrt(-1)
"""
# 输出false 并且有错误
# print(np.sqrt(-1) == np.emath.sqrt(-1))


"""
33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

怎样获取昨天 今天 和明天的时间
"""

# datetime64 知识点

yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')

print(yesterday)

print(today)

print(tomorrow)

print("*********************************")

"""
34. How to get all the dates corresponding to the month of July 2016? (★★☆)

如何获取 2016年 月所有的日期值
"""

z_1 = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(z_1)

print("*********************************")

"""
35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆)

对这个算式进行计算
不能进行拷贝
要原地进行计算

"""

A = np.ones(3)*1

print("               A         ")
print(A)
B = np.ones(3)*2
print("               B         ")
print(B)
C = np.ones(3)*3
print("               C         ")
print(C)

np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)
print("*********************************")

"""
36. Extract the integer part of a random array using 5 different methods (★★☆)

用五种不同的方法从随机数组里面抽取整数部分
"""

z_2 = np.random.uniform(0,10,10)
print("                      z_2")
print(z_2)
print (z_2 - z_2%1)
print (np.floor(z_2))
print (np.ceil(z_2)-1)
print (z_2.astype(int))
print (np.trunc(z_2))
print("*********************************")

"""
#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)

创建一个 5*5 的矩阵
每行的数值为 0 到 4

"""

z_3 = np.zeros((5,5))
z_3 += np.arange(5)
print(z_3)
print("*********************************")



"""
#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)

设置一个生成函数 
它可以生成十个整数并且把它们用来构建一个数组
"""

def generate():
    for x in range(10):
        yield x
# fromiter知识点

"""
        def fromiter(iterable, dtype, count=-1): # real signature unknown; restored from __doc__
            从可迭代对象创建一个新的一维数组。
        Parameters
        ----------
        iterable : iterable object
            An iterable object providing data for the array.
        dtype : data-type
            The data-type of the returned array.
        count : int, optional
            The number of items to read from *iterable*.  The default is -1,
            which means all data is read.

        Examples
        --------
        >>> iterable = (x*x for x in range(5))
        >>> np.fromiter(iterable, float)
        array([  0.,   1.,   4.,   9.,  16.])

"""
z_4 = np.fromiter(generate(),dtype=float,count=-1)
print(z_4)
print("*********************************")


"""
39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)

生成一个大小为10的向量
数值范围是 0 到 1 （0和1不包括）

"""

# linspace知识点

"""
        def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
             axis=0):
             
             返回指定间隔内的等间隔数字。
             
        返回在间隔[`start`，`stop`]中计算出的num个均匀间隔的样本。
        
        间隔的端点可以选择排除。
        
        
        Parameters
    ----------
    start : array_like
        The starting value of the sequence.
    stop : array_like
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (`samples`, `step`), where `step` is the spacing
        between samples.
    dtype : dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
        
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.
        (结果中的轴用于存储样本。仅当start或stop类似于数组时才相关。默认情况下（0），
        样本将沿着在开始处插入的新轴。使用-1获得一个轴的末端。)
        
         Examples
    --------
    >>> np.linspace(2.0, 3.0, num=5)
    array([2.  , 2.25, 2.5 , 2.75, 3.  ])
    >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
    array([2. ,  2.2,  2.4,  2.6,  2.8])
    >>> np.linspace(2.0, 3.0, num=5, retstep=True)
    (array([2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 8
    >>> y = np.zeros(N)
    >>> x1 = np.linspace(0, 10, N, endpoint=True)
    >>> x2 = np.linspace(0, 10, N, endpoint=False)
    >>> plt.plot(x1, y, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(x2, y + 0.5, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.ylim([-0.5, 1])
    (-0.5, 1)
    >>> plt.show()
    
    
 
        
"""
z_5 = np.linspace(0,1,11,endpoint=False)[1:]
print(z_5)
print("*********************************")


"""
#### 40. Create a random vector of size 10 and sort it (★★☆)

生成一个大小为 10 数值随机的向量 并且进行排序
"""

z_6 = np.random.random(10)
print(z_6)
z_6.sort()
print(z_6)

print("*********************************")
