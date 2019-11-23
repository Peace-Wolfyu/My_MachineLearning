# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/23 13:33'

import matplotlib.pyplot as plt
from  sklearn.neighbors import KNeighborsRegressor
import mglearn
from sklearn.model_selection import train_test_split
import numpy as np


X,y = mglearn.datasets.make_wave(n_samples = 40)

# 将数据集分为训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 创建1000个数据点 在-3和3之间均匀分布
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):

    # 利用1个 3个 或9个邻居分别进行预测
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)

    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)

    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target",
                "Test data/target"], loc="best")

plt.show()




















