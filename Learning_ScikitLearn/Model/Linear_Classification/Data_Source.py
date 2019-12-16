# -*- coding: utf-8 -*-

# @Time  : 2019/12/16 15:55

# @Author : Mr.Lin


'''

提供数据
'''
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

data_X = cancer.data
data_y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=42)



