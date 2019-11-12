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