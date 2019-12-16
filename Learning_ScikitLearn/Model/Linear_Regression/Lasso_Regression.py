# -*- coding: utf-8 -*-

# @Time  : 2019/12/16 14:18

# @Author : Mr.Lin


'''

除了 Ridge ，还有一种正则化的线性回归是 Lasso 。与岭回归相同，使用 lasso 也是约束系
数使其接近于 0，但用到的方法不同，叫作 L1 正则化。
8 L1 正则化的结果是，使用 lasso 时
某些系数刚好为 0。这说明某些特征被模型完全忽略。这可以看作是一种自动化的特征选
择。某些系数刚好为 0，这样模型更容易解释，也可以呈现模型最重要的特征。


'''


from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from Learning_ScikitLearn.Model.Linear_Regression.DataSource import X_train,X_test,y_train,y_test,data_X,data_y
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd

# lassp_model = Lasso()

def test_alpha_effect():
    alphas = [0.1,1,10]

    for alpha in alphas:
        lassp_model = Lasso(alpha=alpha)

        lassp_model.fit(X_train,y_train)

        y_pre = lassp_model.predict(X_test)

        print("alpha 为: {} 下的MSE ： {} \n".format(alpha,metrics.mean_squared_error(y_test, y_pre)))


# alpha 为: 0.1 下的MSE ： 33.51060310129611
#
# alpha 为: 1 下的MSE ： 34.069594753479606
#
# alpha 为: 10 下的MSE ： 50.63616615613884


# test_alpha_effect()


# Lasso (Least absolute shrinkage and selection operator, Tibshirani(1996)) 方法是一种压缩估计。它通过构造一个罚函数得到一个较为精炼的模型，使得它压缩一些系数，同时设定一些系数为零。因此保留了子集收缩的优点，是一种处理具有复共线性数据的有偏估计。Lasso 在学习过程中可以直接将部分 feature 系数赋值为 0 的特点使其在调参时非常方便。


# 其中，主要关注参数为 alpha，其作用与 sklearn 包下的另一模型 Ridge 类似，是一个衡量模型灵活度的正则参数。正则度越高，越不可能 overfit。但是它也会导致模型灵活度的降低，可能无法捕捉数据中的所有信号。

'''
通过 Lasso 自带的 CV（Cross Validation）设置，可以直接通过机器挑选最好的 alpha。


'''


model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y_train) # 此处 alpha 为通常值 #fit 把数据套进模型里跑

coef =model_lasso.coef_# .coef_ 可以返回经过学习后的所有 feature 的参数。
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")












