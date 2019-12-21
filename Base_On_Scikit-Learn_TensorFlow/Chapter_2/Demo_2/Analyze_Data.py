# -*- coding: utf-8 -*-
# @Time  : 2019/12/21 15:33
# @Author : Mr.Lin

'''

分析数据

'''

from Chapter_2.Demo_2.DataSource import housing

# 查看前五行的数据 每一行代表一个区
print(housing.head())
print("")


# info方法获取数据集的简单描述
print(housing.info())
print("")

# 查看 ocean_proximity 属性有多少中分类存在
print(housing["ocean_proximity"].value_counts())
print("")

print(housing.describe())
print("")


# 绘制直方图

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()



