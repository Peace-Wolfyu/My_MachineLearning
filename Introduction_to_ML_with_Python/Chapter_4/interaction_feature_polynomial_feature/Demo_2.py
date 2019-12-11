# -*- coding: utf-8 -*-

# @Time  : 2019/12/11 12:55

# @Author : Mr.Lin





'''
我们再次观察波士顿房价数据集，作为对交互特征和多项式特征更加实际的应用。我们在
第 2 章已经在这个数据集上使用过多项式特征了。现在来看一下这些特征的构造方式，以
及多项式特征的帮助有多大。首先加载数据，然后利用 MinMaxScaler 将其缩放到 0 和 1
之间：

'''

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split\
(boston.data, boston.target, random_state=0)
# 缩放数据
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



'''
下面我们提取多项式特征和交互特征，次数最高为 2：
'''

poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

'''
X_train.shape: (379, 13)
X_train_poly.shape: (379, 105)


原始数据有 13 个特征，现在被扩展到 105 个交互特征。这些新特征表示两个不同的原始
特征之间所有可能的交互项，以及每个原始特征的平方。这里 degree=2 的意思是，我们需
要由最多两个原始特征的乘积组成的所有特征。利用 get_feature_names 方法可以得到输
入特征和输出特征之间的确切对应关系：
'''
# print("X_train.shape: {}".format(X_train.shape))
# print("X_train_poly.shape: {}".format(X_train_poly.shape))

'''
Polynomial feature names:
['1', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 
'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x0^2', 'x0 x1', 'x0 x2',
 'x0 x3', 'x0 x4', 'x0 x5', 'x0 x6', 'x0 x7', 'x0 x8', 
 'x0 x9', 'x0 x10', 'x0 x11', 'x0 x12', 'x1^2', 'x1 x2', 'x1 x3'
 , 'x1 x4', 'x1 x5', 'x1 x6', 'x1 x7', 'x1 x8', 'x1 x9', 'x1 x10',
  'x1 x11', 'x1 x12', 'x2^2', 'x2 x3', 'x2 x4', 'x2 x5', 'x2 x6', 'x2 x7',
   'x2 x8', 'x2 x9', 'x2 x10', 'x2 x11', 'x2 x12', 'x3^2', 'x3 x4', 'x3 x5',
    'x3 x6', 'x3 x7', 'x3 x8', 'x3 x9', 'x3 x10', 'x3 x11', 'x3 x12', 'x4^2',
     'x4 x5', 'x4 x6', 'x4 x7', 'x4 x8', 'x4 x9', 'x4 x10', 'x4 x11', 'x4 x12', '
     x5^2', 'x5 x6', 'x5 x7', 'x5 x8', 'x5 x9', 'x5 x10', 'x5 x11', 'x5 x12', 'x6^2',
      'x6 x7', 'x6 x8', 'x6 x9', 'x6 x10', 'x6 x11', 'x6 x12', 'x7^2', 'x7 x8', 'x7 x9', 
      'x7 x10', 'x7 x11', 'x7 x12', 'x8^2', 'x8 x9', 'x8 x10', 'x8 x11', 'x8 x12',
       'x9^2', 'x9 x10', 'x9 x11', 'x9 x12', 'x10^2', 'x10 x11', 'x10 x12', 'x11^2', 
       'x11 x12', 'x12^2']
       
       
       
第一个新特征是常数特征，这里的名称是 "1" 。接下来的 13 个特征是原始特征（名称从
"x0" 到 "x12" ）。然后是第一个特征的平方（ "x0^2" ）以及它与其他特征的组合。
       
'''
# print("Polynomial feature names:\n{}".format(poly.get_feature_names()))


'''
对 Ridge 在有交互特征的数据上和没有交互特征的数据上的性能进行对比：
'''
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train_scaled, y_train)

'''
Score without interactions: 0.621
Score with interactions: 0.753

显然，在使用 Ridge 时，交互特征和多项式特征对性能有很大的提升
'''
# print("Score without interactions: {:.3f}".format(
# ridge.score(X_test_scaled, y_test)))
# ridge = Ridge().fit(X_train_poly, y_train)
# print("Score with interactions: {:.3f}".format(
# ridge.score(X_test_poly, y_test)))



'''
如果使用更加复
杂的模型（比如随机森林），情况会稍有不同：
'''

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train)


'''
Score without interactions: 0.792
Score with interactions: 0.772

可以看到，即使没有额外的特征，随机森林的性能也要优于 Ridge 。添加交互特征和多
项式特征实际上会略微降低其性能。

'''
print("Score without interactions: {:.3f}".format(
rf.score(X_test_scaled, y_test)))
rf = RandomForestRegressor(n_estimators=100).fit(X_train_poly, y_train)
print("Score with interactions: {:.3f}".format(rf.score(X_test_poly, y_test)))






















































