# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/13 15:58'


"""
 1 Import the numpy package under the name `np` 
 通用引用别名‘np’导入numpy包
"""

import numpy as np


"""
#### 2. Print the numpy version and the configuration (★☆☆)

打印numpy的版本以及配置
"""

print("numpy 版本")
print(np.__version__)
print("numpy 配置")
np.show_config()


"""
#### 3. Create a null vector of size 10 (★☆☆)

创建一个空的 大小为10 的向量
"""
print("************************************")

# zeros()  知识点
"""
文档：
  zeros(shape, dtype=float, order='C')
根据给定的维度以及数据类型，返回一个全是 0 的数组
参数：
    shape：可以是 int类型 或者 元组（类型为int）
    dtype：数据类型  非必填
    order：C 或者 F 非必填 默认 C
    是否以行优先（C）或列优先（F）的顺序存储多维数据在内存中。

"""
z = np.zeros(10)
print(z)

"""
#### 4.  How to find the memory size of any array (★☆☆)

输出任意数组的内存大小
"""
print("************************************")

z_1 = np.zeros((10,10))
print(z_1)
#  size 知识点  itemsize 知识点

"""
  size:
     数组中元素的个数
     
  Examples
    --------
    >>> x = np.zeros((3, 5, 2), dtype=np.complex128)
    >>> x.size
    30
    >>> np.prod(x.shape)
    30
"""
print("%d bytes" % (z_1.size * z_1.itemsize))


"""
#### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆)

通过命令行获取numpy add 函数的 文档

"""

"""
%run `python -c "import numpy; numpy.info(numpy.add)"`
"""

"""
#### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆)

创建一个空的 大小为10 的向量，但是第五个的值为 1
"""

z_2 = np.zeros(10)

# 索引从 0 开始
z_2[4] = 1

print(z_2)

"""
#### 7.  Create a vector with values ranging from 10 to 49 (★☆☆)

创建一个向量  里面的值 从 10 到 49
"""


print("************************************")

# arange() 知识点
"""
            arange([start,] stop[, step,], dtype=None)
            返回给定间隔内的均匀间隔的值。
            值在半开区间隔``[start，stop）''（换句话说，包括`start`但不包括`stop`）的间隔内生成。
            对于整数参数，该函数等效于Python内置的“ range”函数，但返回 ndarray 而不是list。
            
            
            当使用非整数步骤（例如0.1）时，结果通常将不一致。在这些情况下，最好使用“ numpy.linspace”。 
            
            参数
            start ： 开始位置 非必填 默认为 0 
            stop：结束位置  不包括
            step：步长 非必填
                  值之间的间距。对于任何输出“ out”，
                  这是两个相邻值“ out [i + 1]-out [i]”之间的距离。
                  默认步长为1。如果将step用作位置参数，则还必须指定start。
            dtype
            
             Examples
        --------
        >>> np.arange(3)
        array([0, 1, 2])
        >>> np.arange(3.0)
        array([ 0.,  1.,  2.])
        >>> np.arange(3,7)
        array([3, 4, 5, 6])
        >>> np.arange(3,7,2)
        array([3, 5])
"""
z_3 = np.arange(10,50)

print(z_3)


"""
#### 8.  Reverse a vector (first element becomes last) (★☆☆)

转置一个向量
"""
print("************************************")

z_4 = np.arange(10)

print(z_4)

# [::-1]知识点
z_4 = z_4[::-1]

print(z_4)

"""
#### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)

创建一个 3 * 3 的矩阵  数值设置为 从 0 到 8
"""
print("************************************")

"""
        a.reshape(shape, order='C')

        返回一个数组，其中包含具有相同数据类型的新的形态的集合
        
        与函数numpy.reshape不同，在ndarray上的此方法允许shape参数的元素作为单独的参数传递。
        例如，``a.reshape（（10，11））''等同于``a.reshape（（10，11））''。


 
"""
z_5 = np.arange(9).reshape(3,3)

print(z_5)

"""

10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆)

从 给出列表中找出 不为 0 的 元素的索引

"""
print("************************************")

# nonzero() 知识点

"""
        def nonzero(a):
        返回非零元素的索引。
        返回一个数组元组，其中每个数组对应一个“ a”维度，
        其中包含该维度中非零元素的索引。“ a”中的值始终以C行风格的行测试顺序返回。
        
        Examples
    --------
    >>> x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
    >>> x
    array([[3, 0, 0],
           [0, 4, 0],
           [5, 6, 0]])
    >>> np.nonzero(x)
    (array([0, 1, 2, 2]), array([0, 1, 0, 1]))

    >>> x[np.nonzero(x)]
    array([3, 4, 5, 6])
    >>> np.transpose(np.nonzero(x))
    array([[0, 0],
           [1, 1],
           [2, 0],
           [2, 1]])

    A common use for ``nonzero`` is to find the indices of an array, where
    a condition is True.  Given an array `a`, the condition `a` > 3 is a
    boolean array and since False is interpreted as 0, np.nonzero(a > 3)
    yields the indices of the `a` where the condition is true.
    (``nonzero''的常见用法是查找条件为True的数组的索引。给定一个数组a，条件a> 3是一个布尔数组，
    由于False被解释为0，因此np.nonzero（a> 3）得出条件为true的a的索引。)

    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> a > 3
    array([[False, False, False],
           [ True,  True,  True],
           [ True,  True,  True]])
    >>> np.nonzero(a > 3)
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    Using this result to index `a` is equivalent to using the mask directly:

    >>> a[np.nonzero(a > 3)]
    array([4, 5, 6, 7, 8, 9])
    >>> a[a > 3]  # prefer this spelling
    array([4, 5, 6, 7, 8, 9])

    ``nonzero`` can also be called as a method of the array.

    >>> (a > 3).nonzero()
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
    
    


"""
z_6 = np.nonzero([1,2,0,0,4,0])
print(z_6)




