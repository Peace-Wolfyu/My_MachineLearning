# -*- coding: utf-8 -*-
# @Time  : 2019/12/26 15:03
# @Author : Mr.Lin

"""
学习曲线

"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)  # not shown in the book
    plt.xlabel("Training set size", fontsize=14)  # not shown
    plt.ylabel("RMSE", fontsize=14)  # not shown


# 首先，我们来看训练数据上的性能：当训练
# 集中只包括一两个实例时，模型可以完美拟合，这是为什么曲线是从
# 0开始的。但是，随着新的实例被添加进训练集中，模型不再能完美
# 拟合训练数据了，因为数据有噪声，并且根本就不是线性的。所以训
# 练集的误差一路上升，直到抵达一个高地，从这一点开始，添加新实
# 例到训练集中不再使平均误差上升或下降。然后我们再来看看验证集
# 的性能表现。当训练集实例非常少时，模型不能很好地泛化，这是为
# 什么验证集误差的值一开始非常大，随着模型经历更多的训练数据，
# 它开始学习，因此验证集误差慢慢下降。但是仅靠一条直线终归不能
# 很好地为数据建模，所以误差也停留在了一个高值，跟另一条曲线十
# 分接近。
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
plt.axis([0, 80, 0, 3])  # not shown in the book
plt.show()
