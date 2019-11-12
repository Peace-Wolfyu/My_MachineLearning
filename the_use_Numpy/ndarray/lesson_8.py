# -*- coding: utf-8 -*-
# @Time  : 2019/11/12 20:13
# @Author : Mr.Lin


"""

线性代数

"""


import numpy as np

x = np.array(
    [
    [1.,2.,3.],
    [4.,5.,6.,]
    ]
)


y = np.array(
    [
        [6.,23.],
        [-1,7],
        [8,9]
    ]

)

print(x)

"""
dot用于矩阵乘法
"""
print(x.dot(y))

