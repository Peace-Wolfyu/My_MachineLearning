

"""
求解线性方程组
"""

import numpy as np

# 创建矩阵A
A = np.mat("1 -2 1;0 2 -8;-4 5 9")

print(A)

# 创建数组b
b = np.array([0,8,-9])

print(b)

# 使用solve函数解线性方程组
x = np.linalg.solve(A,b)

print(x)

print("检查" ,np.dot(A,x))

