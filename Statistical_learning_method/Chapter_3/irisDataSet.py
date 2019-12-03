# -*- coding: utf-8 -*-
# @Time  : 2019/12/3 19:48
# @Author : Mr.Lin

"""
sklearn.datasets 中的iris数据集


Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。数据集包含150个数据样本，分为3类，
每类50个数据，每个数据包含4个属性。可通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性
预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。
spal 花萼
petal 花瓣

"""



from sklearn.datasets import load_iris



iris = load_iris() # 返回的是Bunch对象 与字典非常类似 包含键和值

# print(" Keys of dataSet : \n{}".format(iris.keys()))

#  Keys of dataSet :
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])

'''
DESCR键对应的值是数据集的简要说明
'''
# print(iris['DESCR'][:193] + "\n.......")
# .. _iris_dataset:
#
# Iris plants dataset
# --------------------
#
# **Data Set Characteristics:**
#
#     :Number of Instances: 150 (50 in each of three classes)
#     :Number of Attributes: 4 numeric, pre
# .......


'''
target_names 对应要预测的花的品种
'''
# print("target_names : \n{}".format(iris['target_names']))
# target_names :
# ['setosa' 'versicolor' 'virginica']


'''
feature_names 对应一个字符串列表  对每一个特征进行了说明
'''
# print("feature_names : \n {}".format(iris['feature_names']))
# feature_names :
#  ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


'''
数据放在 target 和 data 字段
data里面是花萼长度  花萼宽度  花瓣长度  花瓣宽度
格式为Numpy数组
'''

# print("Type of data : {}".format(type(iris['data'])))
# Type of data : <class 'numpy.ndarray'>


'''
data数组每一行对应一朵花
列代表每朵花的四个测量数据
'''
# print(" Shape of data : {}".format(iris['data'].shape))
'''
数组含有150朵不同的花的测量数据 
'''
# Shape of data: (150, 4)


'''
看看前五朵花的数据
'''

# print("First five rows of data: \n{}".format(iris['data'][:5]))
# First five rows of data:
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]
#  [4.6 3.1 1.5 0.2]
#  [5.  3.6 1.4 0.2]]


'''
target 数组是包含每朵花的品种  也是 Numpy数组
'''

# print("Type of target:\n{}".format(type(iris['target'])))
# Type of target:
# <class 'numpy.ndarray'>


'''
用 0 1 2分别代表不同的品种
'''
# print("Target : \n{}".format(iris['target']))
# Target :
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]
























