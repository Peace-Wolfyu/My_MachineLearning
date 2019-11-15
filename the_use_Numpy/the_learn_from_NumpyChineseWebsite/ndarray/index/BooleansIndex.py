# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/15 14:47'


"""
布尔值索引
"""

import numpy as np

data = np.arange(25).reshape((5,5))
# print(data)

"""
>>>
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]
 [20 21 22 23 24]]
 
"""

boolean_data = data > 20

# print(boolean_data)
"""
>>>
[[False False False False False]
 [False False False False False]
 [False False False False False]
 [False False False False False]
 [False  True  True  True  True]]
"""

# print(data[boolean_data])

"""
>>>

[21 22 23 24]

"""


data1 = np.arange(12).reshape((3,4))

# print(data1)


"""
>>>
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
"""

boolean_1 = np.array([True,False,False])

# print(data1[boolean_1])
"""
>>>

[[0 1 2 3]]
"""


boolean_2 = np.array([True,False,False,True])

# print(data1[boolean_1,boolean_2])

"""
>>>
[0 3]
"""

"""
ix_()函数
"""

"""
ix_ 函数可以合并不同的向量来获得各个n元组的结果。
举个例子，如果你想要计算三个向量两两组合的结果a+b*c，
也就是说要计算∑i=0(ai+∏j=0,k=0bj∗ck)∑i=0(ai+∏j=0,k=0bj∗ck)，
在下面的例子中，a,b,c长度分别为4，3，5，这样算下来，最终的结果应该有60(4*3*5）个。
数据量少的时候可以手工算，如果数据量大的话，ix_函数就排上用场了。

"""
a = np.array([2,3,4,5])
b = np.array([8,5,4])
c = np.array([5,4,6,8,3])
ax,bx,cx = np.ix_(a,b,c)

# print(ax)
"""
>>>
[

 [
 [2]
 ]

 [
 [3]
 ]

 [
 [4]
 ]

 [
 [5]
 ]
 
 ]

"""

# print(bx)

"""
>>>
[[[8]
  [5]
  [4]]]
"""

# print(cx)
"""
>>>
[[[5 4 6 8 3]]]

"""

# print(ax.shape)

"""
>>>
(4, 1, 1)
"""

# print(bx.shape)
"""
>>>
(1, 3, 1)
"""

# print(cx.shape)
"""
>>>
(1, 1, 5)
"""

result = ax+bx*cx

# print(result)
"""
[[[42 34 50 66 26]
  [27 22 32 42 17]
  [22 18 26 34 14]]

 [[43 35 51 67 27]
  [28 23 33 43 18]
  [23 19 27 35 15]]

 [[44 36 52 68 28]
  [29 24 34 44 19]
  [24 20 28 36 16]]

 [[45 37 53 69 29]
  [30 25 35 45 20]
  [25 21 29 37 17]]]
"""

# print(result.shape)

"""
>>>
(4, 3, 5)
"""


# 还可以像下面一样来执行同样的功能：

def ufunc_reduce(ufct, *vectors):
    vs = np.ix_(*vectors)
    r = ufct.identity
    for v in vs:
        r = ufct(r,v)
    return r
# and then use it as:
print(ufunc_reduce(np.add,a,b,c))
















