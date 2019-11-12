# -*- coding: utf-8 -*-
# @Time  : 2019/11/12 20:22
# @Author : Mr.Lin




"""

series 是一种类似于一维数组的对象，它由一组数据以及一组与之相关的数据标签（索引）组成


"""


import pandas as pd


obj = pd.Series(
    [4,9,0,11]
)

print(obj)

"""
可以通过Series 的values以及index属性获取其数组表示形式和索引对象
"""

print(obj.values)


"""
希望对各个数据点进行标记自己的索引
"""

obj2 = pd.Series(
    [4,1,2,3],
    index=['d','c','s','ddd']
)

print(obj2)









