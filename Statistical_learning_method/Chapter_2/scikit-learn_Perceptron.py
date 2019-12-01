# -*- coding: utf-8 -*-

# @Time  : 2019/12/1 17:18

# @Author : Mr.Lin


from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt



iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target


#
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
df.label.value_counts()
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]


clf = Perceptron(fit_intercept=False, n_iter=1000, shuffle=False)
clf.fit(X, y)

# Weights assigned to the features.
print(clf.coef_)

print("")
print("")

# 截距 Constants in decision function.
print(clf.intercept_)


x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)

plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()

plt.show()


















