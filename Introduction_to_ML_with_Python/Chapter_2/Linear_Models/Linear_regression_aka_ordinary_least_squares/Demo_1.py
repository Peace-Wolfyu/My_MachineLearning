# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/23 13:55'



" 线性回归 最小二乘法 "


from sklearn.linear_model import LinearRegression
import mglearn
from sklearn.model_selection import train_test_split



X,y = mglearn.datasets.make_wave(n_samples = 60)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

lr = LinearRegression().fit(X_train,y_train)


print("lr.coef_: ()".format(lr.coef_))


print("")
print("")

print("lr.intercept_: {}".format(lr.intercept_))

'斜率 参数被保存在 coef_属性中' \
'偏移或者截距保存在 intercept_属性中'
# >>>
# lr.coef_: ()
#
#
# lr.intercept_: -0.031804343026759746

# 训练集和测试集性能
print("")
print("")

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# >>>
# Training set score: 0.67
# Test set score: 0.66


















































































