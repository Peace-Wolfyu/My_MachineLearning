# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/12 17:28'


"""
数据处理
"""

import numpy as  np


"""
在一组值上计算函数sqrt(x2 + y2) 

np.meshgrid函数接受两个一维数组

产生两个二维矩阵

"""


points = np.arange(-5,5,0.01)

xs,ys = np.meshgrid(points,points)

print(ys)

z = np.sqrt(xs ** 2 + ys ** 2)

print(z)

"""
将条件逻辑表述为数组运算
"""


"""
假设有一个布尔数组以及两个值数组
"""

xarr = np.array([
    1.1,
    1.2,
    1.3,
    1.4,
    1.5
])

yarr = np.array(
    [
        2.1,
        2.2,
        2.3,
        2.4,
        2.5
    ]
)

cond = np.array([
    True,
    False,
    True,
    True,
    False
])

"""
假设根据cond中的值选取xarr和yarr
当cond中的值为true，选取xarr
当cond中的值为false，选取yarr
"""

"""
使用np.where函数
"""


result = np.where(cond,xarr,yarr)

print(result)


"""
np.where第二个以及第三个参数不必是数组，也可以是标量值

where用于根据另一个数组而产生一个新的数组

假设有一个由随机数据组成的矩阵，希望将所有正值替换成2
所有负值替换成-2


"""

arr = np.random.rand(4,4)

print(arr)


print(arr > 0)

print(np.where(arr > 0 ,2 ,2))
















