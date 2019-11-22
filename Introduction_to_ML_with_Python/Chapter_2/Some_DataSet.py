# -*- coding: utf-8 -*-
# @Time  : 2019/11/22 20:22
# @Author : Mr.Lin



"模拟的二分类数据 forge数据集 两个特征"

' 绘制散点图 '

import matplotlib.pyplot as plt
import mglearn

X,y = mglearn.datasets.make_forge()

# mglearn.discrete_scatter(X[:,0],X[:,1],y)
#
# plt.legend(["Class 0","Class 1"],loc=4)
#
# plt.xlabel("First Feature")
# plt.ylabel("Second Feature")
print("X.shape: {}".format(X.shape))

# plt.show()

' 从 X.shape: (26, 2)可以看出数据集 包含 26个数据点以及2个特征 '


" 用模拟的wave数据集来说明回归算法" \
"wave数据集只有一个输入特征以及一个连续的目标变量 后者是模型想要预测的对象"


X_1,y_1 = mglearn.datasets.make_wave(n_samples=40)

plt.plot(X_1,y_1,'o')

plt.ylim(-3,3)

plt.xlabel("Feature")

plt.ylabel("Target")

' x轴表示特征  y轴表示回归目标'
plt.show()





























































