# -*- coding: utf-8 -*-
# @Time  : 2019/12/22 14:29
# @Author : Mr.Lin


'''



'''
from Chapter_3.Demo_3.data_source import mnist
import numpy as np
X, y = mnist["data"], mnist["target"]
# MNIST数据集已经分成训练集（前6万张图像）和测试集（最后1万张图像）
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# 将训练集数据洗牌
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
