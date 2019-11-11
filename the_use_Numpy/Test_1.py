#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/11 19:22
# @File    : Test_1.py

"""
矩阵运算
"""
import numpy as np

#初始化一个矩阵
A = np.mat("0 1 2;1 0 3;4 3 7")

print(A)

# 使用inv函数计算逆矩阵

inverse = np.linalg.inv(A)

print(inverse)


# 检查是否为逆矩阵

print( A * inverse)

