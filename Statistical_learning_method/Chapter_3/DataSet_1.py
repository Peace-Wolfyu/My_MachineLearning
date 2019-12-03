# -*- coding: utf-8 -*-
# @Time  : 2019/12/3 20:31
# @Author : Mr.Lin


from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
"""
缩短数据集 方便观察
"""

iris = load_iris() # 返回的是Bunch对象 与字典非常类似 包含键和值
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

plt.scatter(df[0:10]['sepal length'], df[:10]['sepal width'], label='0')
plt.scatter(df[70:80]['sepal length'], df[70:80]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
# plt.show()

data1 = np.array(df.iloc[0:10, [0, 1, -1]])
data2 = np.array(df.iloc[70:80, [0, 1, -1]])
data = np.append(data1,data2,axis=0)
# print("Data : \n{}".format(data))
# print(data.shape)

# (20, 3)

# Data :
# [[5.1 3.5 0. ]
#  [4.9 3.  0. ]
#  [4.7 3.2 0. ]
#  [4.6 3.1 0. ]
#  [5.  3.6 0. ]
#  [5.4 3.9 0. ]
#  [4.6 3.4 0. ]
#  [5.  3.4 0. ]
#  [4.4 2.9 0. ]
#  [4.9 3.1 0. ]
#  [5.9 3.2 1. ]
#  [6.1 2.8 1. ]
#  [6.3 2.5 1. ]
#  [6.1 2.8 1. ]
#  [6.4 2.9 1. ]
#  [6.6 3.  1. ]
#  [6.8 2.8 1. ]
#  [6.7 3.  1. ]
#  [6.  2.9 1. ]
#  [5.7 2.6 1. ]]




































