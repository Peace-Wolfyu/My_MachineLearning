# -*- coding: utf-8 -*-
# @Time  : 2019/11/15 20:44
# @Author : Mr.Lin


import numpy as np


"""
数组计算
基本数学计算函数会对数组中元素逐个进行计算，既可以利用操作符重载，也可以使用函数方式：
"""

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
print(np.add(x, y))
# >>>
# [[ 6.  8.]
#  [10. 12.]]
# [[ 6.  8.]
#  [10. 12.]]
print("")
print("")

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(x - y)
print(np.subtract(x, y))
# >>>
# [[-4. -4.]
#  [-4. -4.]]
# [[-4. -4.]
#  [-4. -4.]]
print("")
print("")
# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x * y)
print(np.multiply(x, y))
# >>>
# [[ 5. 12.]
#  [21. 32.]]
# [[ 5. 12.]
#  [21. 32.]]
print("")
print("")

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))




















































































































