# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/12 16:33'

"""
正式开始Numpy的learning

"""
import numpy as np

"""
创建ndarray
"""

# 给定一个列表
data1 = [6,7.5,8,0,1]

"""
使用array函数
接受一切序列型的对象，产生一个新的ndarray
"""
arr1 = np.array(data1)

print(arr1)

"""
每个数组都有一个shape和一个dtype

shape：表示数组各维度大小的元组

dtype：说明数组数据类型的对象

"""

print(arr1.shape)

print(arr1.dtype)



"""
嵌套序列
"""

data2 = [
    [1,2,3,4],
    [4,5,6,7]
]

arr2 = np.array(data2)

print(data2)

"""
数组的运算
"""

arr = np.array([
    [1,2,3],
    [4,5,6]
])

print(arr)

print(arr.shape)

# 乘法运算
opention1 = arr * arr

print(opention1)


opention2 = 1 / arr

print(opention2)