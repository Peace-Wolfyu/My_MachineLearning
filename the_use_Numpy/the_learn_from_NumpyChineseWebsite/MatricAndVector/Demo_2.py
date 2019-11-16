# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 16:39'

import numpy as np

"""

numpy求解方程组
线性代数中比较常见的问题之一是求解矩阵向量方程。 这是一个例子。 我们寻找解决方程的向量x

Ax = b



"""
A = np.array([[2,1,-2],[3,0,1],[1,1,-1]])
b = np.transpose(np.array([[-3,5,-2]]))


x = np.linalg.solve(A,b)

# print(x)
# >>>
# [[ 1.]
#  [-1.]
#  [ 2.]]

































































