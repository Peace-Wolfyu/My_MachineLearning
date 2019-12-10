# -*- coding: utf-8 -*-
# @Time  : 2019/12/10 20:32
# @Author : Mr.Lin

'''

数据表示的最佳方法不仅取决于数据的语义，还取决于所使用的模型种类。线性模型与基
于树的模型（比如决策树、梯度提升树和随机森林）是两种成员很多同时又非常常用的模
型，它们在处理不同的特征表示时就具有非常不同的性质。我们回到第 2 章用过的 wave 回
归数据集。它只有一个输入特征。下面是线性回归模型与决策树回归在这个数据集上的对
比
'''

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import mglearn
import numpy as np
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)


'''
正如你所知，线性模型只能对线性关系建模，对于单个特征的情况就是直线。决策树可以
构建更为复杂的数据模型，但这强烈依赖于数据表示。有一种方法可以让线性模型在连续
数据上变得更加强大，就是使用特征分箱（binning，也叫离散化，即 discretization）将其
划分为多个特征，如下所述。
'''
# reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
# plt.plot(line, reg.predict(line), label="decision tree")
# reg = LinearRegression().fit(X, y)
# plt.plot(line, reg.predict(line), label="linear regression")
# plt.plot(X[:, 0], y, 'o', c='k')
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")
# plt.legend(loc="best")
# plt.show()



'''
我们假设将特征的输入范围（在这个例子中是从 -3 到 3）划分成固定个数的箱子（bin），
比如 10 个，那么数据点就可以用它所在的箱子来表示。为了确定这一点，我们首先需
要定义箱子。在这个例子中，我们在 -3 和 3 之间定义 10 个均匀分布的箱子。我们用
np.linspace 函数创建 11 个元素，从而创建 10 个箱子，即两个连续边界之间的空间：
'''

bins = np.linspace(-3,3,11)
'''
bins :[-3.  -2.4 -1.8 -1.2 -0.6  0.   0.6  1.2  1.8  2.4  3. ]

'''
# print("bins :{}".format(bins))

'''
这里第一个箱子包含特征取值在 -3 到 -2.4 之间的所有数据点，第二个箱子包含特征取值
在 -2.4 到 -1.8 之间的所有数据点，以此类推。
接下来，我们记录每个数据点所属的箱子。这可以用 np.digitize 函数轻松计算出来：
'''

which_bin = np.digitize(X, bins=bins)
'''
Data points:
 [[-0.75275929]
 [ 2.70428584]
 [ 1.39196365]
 [ 0.59195091]
 [-2.06388816]]

Bin membership for data points:
 [[ 4]
 [10]
 [ 8]
 [ 6]
 [ 2]]
'''

# print("\nData points:\n", X[:5])
# print("\nBin membership for data points:\n", which_bin[:5])

'''
我们在这里做的是将 wave 数据集中单个连续输入特征变换为一个分类特征，用于表示数
据点所在的箱子。要想在这个数据上使用 scikit-learn 模型，我们利用 preprocessing 模
块的 OneHotEncoder 将这个离散特征变换为 one-hot 编码。 OneHotEncoder 实现的编码与
pandas.get_dummies 相同，但目前它只适用于值为整数的分类变量：
'''
from sklearn.preprocessing import OneHotEncoder

# 使用OneHotEncoder进行变换
encoder = OneHotEncoder(sparse=False)
# encoder.fit找到which_bin中的唯一值
encoder.fit(which_bin)
# transform创建one-hot编码
X_binned = encoder.transform(which_bin)
'''
[[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]
'''
'''
由于我们指定了 10 个箱子，所以变换后的 X_binned 数据集现在包含 10 个特征：
'''
# print(X_binned[:5])

'''
X_binned.shape: (100, 10)
'''
print("X_binned.shape: {}".format(X_binned.shape))



'''
下面我们在 one-hot 编码后的数据上构建新的线性模型和新的决策树模型。结果见图 4-2，
箱子的边界由黑色虚线表示：
'''
line_binned = encoder.transform(np.digitize(line, bins=bins))
reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='linear regression binned')
reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='decision tree binned')
plt.plot(X[:, 0], y, 'o', c='k')
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.legend(loc="best")
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.show()


'''
虚线和实线完全重合，说明线性回归模型和决策树做出了完全相同的预测。对于每个箱
子，二者都预测一个常数值。因为每个箱子内的特征是不变的，所以对于一个箱子内的所
有点，任何模型都会预测相同的值。比较对特征进行分箱前后模型学到的内容，我们发
现，线性模型变得更加灵活了，因为现在它对每个箱子具有不同的取值，而决策树模型的
灵活性降低了。分箱特征对基于树的模型通常不会产生更好的效果，因为这种模型可以学
习在任何位置划分数据。从某种意义上来看，决策树可以学习如何分箱对预测这些数据最
为有用。此外，决策树可以同时查看多个特征，而分箱通常针对的是单个特征。不过，线
性模型的表现力在数据变换后得到了极大的提高。
对于特定的数据集，如果有充分的理由使用线性模型——比如数据集很大、维度很高，但
有些特征与输出的关系是非线性的——那么分箱是提高建模能力的好方法。
'''













































