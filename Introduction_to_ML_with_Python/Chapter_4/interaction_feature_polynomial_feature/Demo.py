# -*- coding: utf-8 -*-
# @Time  : 2019/12/10 20:47
# @Author : Mr.Lin

import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X, y = mglearn.datasets.make_wave(n_samples=100)
bins = np.linspace(-3,3,11)

from sklearn.preprocessing import OneHotEncoder
which_bin = np.digitize(X, bins=bins)

# 使用OneHotEncoder进行变换
encoder = OneHotEncoder(sparse=False)
# encoder.fit找到which_bin中的唯一值
encoder.fit(which_bin)
# transform创建one-hot编码
X_binned = encoder.transform(which_bin)

'''
想要丰富特征表示，特别是对于线性模型而言，另一种方法是添加原始数据的交互特征
（interaction feature）和多项式特征（polynomial feature）。这种特征工程通常用于统计建模，
但也常用于许多实际的机器学习应用中。
作为第一个例子，我们再看一次图 4-2。线性模型对 wave 数据集中的每个箱子都学到一个
常数值。但我们知道，线性模型不仅可以学习偏移，还可以学习斜率。想要向分箱数据上
的线性模型添加斜率，一种方法是重新加入原始特征（图中的 x 轴）。这样会得到 11 维的
数据集，
'''
X_combined = np.hstack([X, X_binned])
'''
(100, 11)

'''
print(X_combined.shape)

reg = LinearRegression().fit(X_combined, y)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
line_binned = encoder.transform(np.digitize(line, bins=bins))

line_combined = np.hstack([line, line_binned])
# plt.plot(line, reg.predict(line_combined), label='linear regression combined')
# for bin in bins:
#     plt.plot([bin, bin], [-3, 3], ':', c='k')
# plt.legend(loc="best")
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")
# plt.plot(X[:, 0], y, 'o', c='k')
'''
在这个例子中，模型在每个箱子中都学到一个偏移，还学到一个斜率。学到的斜率是向下
的，并且在所有箱子中都相同——只有一个 x 轴特征，也就只有一个斜率。因为斜率在所
有箱子中是相同的，所以它似乎不是很有用。我们更希望每个箱子都有一个不同的斜率！
为了实现这一点，我们可以添加交互特征或乘积特征，用来表示数据点所在的箱子以及数
据点在 x 轴上的位置。这个特征是箱子指示符与原始特征的乘积。我们来创建数据集：
'''
# plt.show()

X_product = np.hstack([X_binned, X * X_binned])
'''
(100, 20)

'''
print(X_product.shape)


'''
这个数据集现在有 20 个特征：数据点所在箱子的指示符与原始特征和箱子指示符的乘积。
你可以将乘积特征看作每个箱子 x 轴特征的单独副本。它在箱子内等于原始特征，在其他
位置等于零。
'''
# reg = LinearRegression().fit(X_product, y)
# line_product = np.hstack([line_binned, line * line_binned])
# plt.plot(line, reg.predict(line_product), label='linear regression product')
# for bin in bins:
#     plt.plot([bin, bin], [-3, 3], ':', c='k')
# plt.plot(X[:, 0], y, 'o', c='k')
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")
# plt.legend(loc="best")
'''
现在这个模型中每个箱子都有自己的偏移和斜率。
使 用 分 箱 是 扩 展 连 续 特 征 的 一 种 方 法。 另 一 种 方 法 是 使 用 原 始 特 征 的 多 项 式
（polynomial）。对于给定特征 x ，我们可以考虑 x ** 2 、 x ** 3 、 x ** 4 ，等等。这在
preprocessing 模块的 PolynomialFeatures 中实现：
'''
# plt.show()

from sklearn.preprocessing import PolynomialFeatures
# 包含直到x ** 10的多项式:
# 默认的"include_bias=True"添加恒等于1的常数特征
poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)
'''
多项式的次数为 10，因此生成了 10 个特征：

X_poly.shape: (100, 10)
'''
print("X_poly.shape: {}".format(X_poly.shape))
print("")

'''
比较 X_poly 和 X 的元素：
'''

'''
Entries of X:
[[-0.75275929]
 [ 2.70428584]
 [ 1.39196365]
 [ 0.59195091]
 [-2.06388816]]

Entries of X_poly:
[[-7.52759287e-01  5.66646544e-01 -4.26548448e-01  3.21088306e-01
  -2.41702204e-01  1.81943579e-01 -1.36959719e-01  1.03097700e-01
  -7.76077513e-02  5.84199555e-02]
 [ 2.70428584e+00  7.31316190e+00  1.97768801e+01  5.34823369e+01
   1.44631526e+02  3.91124988e+02  1.05771377e+03  2.86036036e+03
   7.73523202e+03  2.09182784e+04]
 [ 1.39196365e+00  1.93756281e+00  2.69701700e+00  3.75414962e+00
   5.22563982e+00  7.27390068e+00  1.01250053e+01  1.40936394e+01
   1.96178338e+01  2.73073115e+01]
 [ 5.91950905e-01  3.50405874e-01  2.07423074e-01  1.22784277e-01
   7.26822637e-02  4.30243318e-02  2.54682921e-02  1.50759786e-02
   8.92423917e-03  5.28271146e-03]
 [-2.06388816e+00  4.25963433e+00 -8.79140884e+00  1.81444846e+01
  -3.74481869e+01  7.72888694e+01 -1.59515582e+02  3.29222321e+02
  -6.79478050e+02  1.40236670e+03]]
'''
# print("Entries of X:\n{}".format(X[:5]))
print("")
# print("Entries of X_poly:\n{}".format(X_poly[:5]))


'''
可以通过调用 get_feature_names 方法来获取特征的语义，给出每个特征的指数：
'''

'''
Polynomial feature names:
['x0', 'x0^2', 'x0^3', 'x0^4', 'x0^5', 'x0^6', 'x0^7', 'x0^8', 'x0^9', 'x0^10']
'''
print("Polynomial feature names:\n{}".format(poly.get_feature_names()))


'''
可以看到， X_poly 的第一列与 X 完全对应，而其他列则是第一列的幂。有趣的是，你可
以发现有些值非常大。第二行有大于 20 000 的元素，数量级与其他行都不相同。
将多项式特征与线性回归模型一起使用，可以得到经典的多项式回归（polynomial
regression）模型
'''

'''
多项式特征在这个一维数据上得到了非常平滑的拟合。但高次多项式在边界上
或数据很少的区域可能有极端的表现。
作为对比，下面是在原始数据上学到的核 SVM 模型，没有做任何变换
'''
# reg = LinearRegression().fit(X_poly, y)
# line_poly = poly.transform(line)
# plt.plot(line, reg.predict(line_poly), label='polynomial linear regression')
# plt.plot(X[:, 0], y, 'o', c='k')
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")
# plt.legend(loc="best")
# plt.show()


'''
使用更加复杂的模型（即核 SVM），我们能够学到一个与多项式回归的复杂度类似的预测
结果，且不需要进行显式的特征变换
'''
from sklearn.svm import SVR
for gamma in [1, 10]:
    svr = SVR(gamma=gamma).fit(X, y)
    plt.plot(line, svr.predict(line), label='SVR gamma={}'.format(gamma))
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
# plt.show()






















