# -*- coding: utf-8 -*-
# @Time  : 2019/11/13 19:14
# @Author : Mr.Lin

import numpy as np


"""
31. How to ignore all numpy warnings (not recommended)? (★☆☆)
怎样忽略numpy的警告（这是不推荐的做法）
"""

# Suicide mode on
# defaults = np.seterr(all="ignore")
# Z = np.ones(1) / 0

# Back to sanity
# _ = np.seterr(**defaults)



"""
32. Is the following expressions true? (★☆☆)

np.sqrt(-1) == np.emath.sqrt(-1)
"""
# 输出false 并且有错误
# print(np.sqrt(-1) == np.emath.sqrt(-1))


"""
33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

怎样获取昨天 今天 和明天的时间
"""

# datetime64 知识点

yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')

print(yesterday)

print(today)

print(tomorrow)

print("*********************************")

"""
34. How to get all the dates corresponding to the month of July 2016? (★★☆)

如何获取 2016年 月所有的日期值
"""

z_1 = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(z_1)

print("*********************************")

"""
35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆)

对这个算式进行计算
不能进行拷贝
要原地进行计算

"""

A = np.ones(3)*1

print("               A         ")
print(A)
B = np.ones(3)*2
print("               B         ")
print(B)
C = np.ones(3)*3
print("               C         ")
print(C)

np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)
print("*********************************")

"""
36. Extract the integer part of a random array using 5 different methods (★★☆)

用五种不同的方法从随机数组里面抽取整数部分
"""

z_2 = np.random.uniform(0,10,10)
print("                      z_2")
print(z_2)
print (z_2 - z_2%1)
print (np.floor(z_2))
print (np.ceil(z_2)-1)
print (z_2.astype(int))
print (np.trunc(z_2))
print("*********************************")

"""
#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)

创建一个 5*5 的矩阵
每行的数值为 0 到 4

"""

z_3 = np.zeros((5,5))
z_3 += np.arange(5)
print(z_3)
print("*********************************")



"""
#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)

设置一个生成函数 
它可以生成十个整数并且把它们用来构建一个数组
"""

def generate():
    for x in range(10):
        yield x
# fromiter知识点
z_4 = np.fromiter(generate(),dtype=float,count=-1)
print(z_4)
print("*********************************")


"""
39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)

生成一个大小为10的向量
数值范围是 0 到 1 （0和1不包括）

"""

# linspace知识点
z_5 = np.linspace(0,1,11,endpoint=False)[1:]
print(z_5)
print("*********************************")


"""
#### 40. Create a random vector of size 10 and sort it (★★☆)

生成一个大小为 10 数值随机的向量 并且进行排序
"""

z_6 = np.random.random(10)
print(z_6)
z_6.sort()
print(z_6)

print("*********************************")
