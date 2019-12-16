# -*- coding: utf-8 -*-

# @Time  : 2019/12/16 13:41

# @Author : Mr.Lin



'''

提供数据API

'''


from sklearn import datasets
from sklearn.model_selection import train_test_split


loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2)



