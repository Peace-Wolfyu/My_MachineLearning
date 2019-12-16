# -*- coding: utf-8 -*-

# @Time  : 2019/12/16 13:06

# @Author : Mr.Lin


'''

线性回归预测(波士顿房价数据集（Boston）)
'''






'''
运行线性模型。我们选用sklearn中基于最小二乘的线性回归模型，并用训练集进行拟合，得到拟合直线y=wTx+b中的权重参数w和b：
'''
from sklearn.linear_model import LinearRegression
from Learning_ScikitLearn.Model.Linear_Regression.DataSource import X_train,X_test,y_train,y_test,data_X,data_y
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_predict


model = LinearRegression()
model.fit(X_train, y_train)

# [-7.02849177e-02  5.60724720e-02  4.31044784e-02  2.55220048e+00
#  -1.77516265e+01  3.98583389e+00 -3.06547198e-03 -1.53905615e+00
#   3.11428771e-01 -1.22520393e-02 -8.98555297e-01  1.21071074e-02
#  -5.46723404e-01]

'''
系数矩阵
'''
print (model.coef_)

print("")
print("")


# 33.64897125222605
'''
截距
'''
print (model.intercept_)

'''
模型测试。利用测试集得到对应的结果，并利用均方根误差（MSE）对测试结果进行评价：
'''


# 预测出的房价
y_pred = model.predict(X_test)
print("")
print("")
print("y_pred:\n{}".format(y_pred))
print("")
print("")
print("")
print("")

# MSE: 24.350103774459484

print("MSE:", metrics.mean_squared_error(y_test, y_pred))


'''
交叉验证。我们使用10折交叉验证，即cv=10，并求出交叉验证得到的MSE值


'''
predicted = cross_val_predict(model, data_X, data_y, cv=10)

print("")
print("")

# MSE: 34.59704255768227

print("MSE:", metrics.mean_squared_error(data_y, predicted))


plt.scatter(data_y, predicted, color='y', marker='o')
plt.scatter(data_y, data_y,color='g', marker='+')
plt.show()








































