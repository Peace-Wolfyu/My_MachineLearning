# -*- coding: utf-8 -*-
# @Time  : 2019/11/21 22:40
# @Author : Mr.Lin

"""

    第一个应用：
        鸢尾花分类





"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
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
print("")
print("")

" 训练数据与测试数据"

'scikit-learn 用train_test_split 可以打乱数据集并进行拆分 将' \
'75%的行数据及其对应标签作为训练集  剩下25%数据及其标签作为测试集'

' scikit-learn 中数据通常用大写X表示  标签用小写y表示'


X_train,X_test,y_train,y_test = train_test_split(
    iris_dataset['data'],
    iris_dataset['target'],
    random_state=0
)
' random_state 参数指定了随机数生成器的种子  这样函数的输出就是固定不变的  所以代码的输出始终一样'

print(" X_train shape : {}".format(X_train.shape))
print(" y_train shape : {}".format(y_train.shape))
# >>>
# X_train shape : (112, 4)
#  y_train shape : (112,)

print("")
print("")



print(" X_test shape : {}".format(X_test.shape))
print(" y_test shape : {}".format(y_test.shape))
# >>>
# X_test shape : (38, 4)
#  y_test shape : (38,)

print("")
print("")

" 观察数据 "

' 绘制散点图 '

iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset.feature_names)

pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=
                        .8,cmap=mglearn.cm3)


# pyplot.show()


" 构建第一个模型 "

'k 近邻算法'

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

print(knn.fit(X_train,y_train))

# >>>
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#                      metric_params=None, n_jobs=None, n_neighbors=1, p=2,
#                      weights='uniform')



print("")
print("")


" 做出预测 "

X_new = np.array([
    [5,2.9,1,0.2]
])

print("X_new shape : ()",format(X_new.shape))
# >>>
# X_new shape : () (1, 4)


print("")
print("")















