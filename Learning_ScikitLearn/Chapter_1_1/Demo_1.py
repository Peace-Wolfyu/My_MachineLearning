# -*- coding: utf-8 -*-

# @Time  : 2019/12/15 13:34

# @Author : Mr.Lin


"""

讲述一些用于回归的方法，其中目标值 y 是输入变量 x 的线性组合。 数学概念表示为：如果 \hat{y} 是预测值，那么有：

\hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p

在整个模块中，我们定义向量 w = (w_1,..., w_p) 作为 coef_ ，定义 w_0 作为 intercept_ 。
"""

'''
1.1.1. 普通最小二乘法
'''

'''
LinearRegression
 
拟合一个带有系数 w = (w_1, ..., w_p) 的线性模型，
使得数据集实际观测数据和预测数据（估计值）之间的残差平方和最小。
'''
'''
LinearRegression 会调用 fit 方法来拟合数组 X， y，
并且将线性模型的系数 w 存储在其成员变量 coef_ 中
'''


from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

'''
[0.5 0.5]

'''
print(reg.coef_)

'''
然而，对于普通最小二乘的系数估计问题，其依赖于模型各项的相互独立性。
当各项是相关的，且设计矩阵 X 的各列近似线性相关，那么，设计矩阵会趋向于奇异矩阵，
这种特性导致最小二乘估计对于随机误差非常敏感，可能产生很大的方差。例如，在没有实验设计的情况下收集到的数据，这种多重共线性（multicollinearity）的情况可能真的会出现。
'''










































