# -*- coding: utf-8 -*-

# @Time  : 2019/12/15 13:49

# @Author : Mr.Lin


'''
1.1.2. 岭回归
'''


'''
Ridge 回归通过对系数的大小施加惩罚来解决 普通最小二乘法 的一些问题。 岭系数最小化的是带罚项的残差平方和，

nderset{w}{min\,} {{|| X w - y||_2}^2 + \alpha {||w||_2}^2}

其中， \alpha \geq 0 是控制系数收缩量的复杂性参数： \alpha 的值越大，收缩量越大，模型对共线性的鲁棒性也更强。
'''

'''
与其他线性模型一样， Ridge 用 fit 方法完成拟合，并将模型系数 w 存储在其 coef_ 成员中:
'''

'''
岭回归的复杂度
这种方法与 普通最小二乘法 的复杂度是相同的.
'''


from sklearn import linear_model
reg = linear_model.Ridge (alpha = .5)
reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
# Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
#  normalize=False, random_state=None, solver='auto', tol=0.001)

'''
[0.34545455 0.34545455]

'''
print(reg.coef_)
print("")
'''
0.1363636363636364

'''
print(reg.intercept_)
































































































