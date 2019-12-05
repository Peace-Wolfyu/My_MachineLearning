# -*- coding: utf-8 -*-
# @Time  : 2019/12/5 22:41
# @Author : Mr.Lin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math
from Statistical_learning_method.Chapter_4.MyNaiveBayes import NaiveBayes

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:,:-1], data[:,-1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


naiveBayes = NaiveBayes()

naiveBayes.fit(X_train,y_train)

print(naiveBayes.prediction([4.4,  3.2,  1.3,  0.2]))







