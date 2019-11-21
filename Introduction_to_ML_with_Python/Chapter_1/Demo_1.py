# -*- coding: utf-8 -*-
# @Time  : 2019/11/21 22:40
# @Author : Mr.Lin

"""

    第一个应用：
        鸢尾花分类





"""
from sklearn.datasets import load_iris

'load_iris 返回的iris对象是一个Bunch对象 与字典相似 包含键和值'
' Bunch 对象'
iris_dataset = load_iris()

print(" Keys of iris_dataset : \n {}".format(iris_dataset.keys()))
print("")
print("")
# >>>
# Keys
# of
# iris_dataset:
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])

' DESCR 键对应的值是数据集的简要说明 '
print(iris_dataset['DESCR'][:193] + "\n....")
# >>>
# .. _iris_dataset:
#
# Iris plants dataset
# --------------------
#
# **Data Set Characteristics:**
#
#     :Number of Instances: 150 (50 in each of three classes)
#     :Number of Attributes: 4 numeric, pre
# ....
print("")
print("")

'target_names 键对应的值是一个字符串数组 包含要预测的花的品种'
print("Target name: {}".format(iris_dataset['target_names']))
print("")
print("")
# >>>
# Target name: ['setosa' 'versicolor' 'virginica']

' feature_names 键对应值是一个字符串列表 对每一个特征进行了说明'
print("Feature names : \n{}".format(iris_dataset['feature_names']))
print("")
print("")
# >>>
# Feature names :
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

' 数据包含在 target 和 data 字段中 ' \
'data 里面是花萼长度 宽度   花瓣长度 宽度 的测量数据  格式为Numpy 数组'

print("Type of data: {}".format(type(iris_dataset['data'])))
# >>>
# Type of data: <class 'numpy.ndarray'>

print("")
print("")


' data 数组的每一行对应一朵花  列代表每朵花的四个测量数据'

print(" Shape of data : {}".format(iris_dataset['data'].shape))
print("")
print("")
# >>>
# Shape
# of
# data: (150, 4)

' 看出数组有150朵不同的花的测量数据 '
' data数组的形状（shape）是样本数 * 特征数'

print(" First five rows of data : \n{}".format(iris_dataset['data'][:5]))
# >>>
#  First five rows of data :
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]
#  [4.6 3.1 1.5 0.2]
#  [5.  3.6 1.4 0.2]]

' 前五朵花的具体数据'
print("")
print("")

' target 数组包含的测量过的每朵花的品种  也是Numpy数组'

print(" Type of target : {}".format(type(iris_dataset['target'])))

# >>>
# Type
# of
# target: <class 'numpy.ndarray'>

print("")
print("")

' target 一维数组 每朵花对应其中一个数据'

print("Shape of target : {}".format(iris_dataset['target'].shape))
# >>>
# Shape of target : (150,)


print("")
print("")

'品种被转换成从 0 到 2 的整数'

print(" Target :\n{}".format(iris_dataset['target']))
# >>>
#  Target :
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]


' 0 ： setosa'
' 1 : versicolor'
' 2 : virginica'




































