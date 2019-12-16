# -*- coding: utf-8 -*-
# @Time  : 2019/12/16 19:09
# @Author : Mr.Lin
from sklearn import metrics
from sklearn.svm import LinearSVC

from Learning_ScikitLearn.Model.Linear_Classification.Data_Source import X_test,X_train,y_train,y_test,data_y,data_X



linear_svc = LinearSVC()

# 训练数据
linear_svc.fit(X_train,y_train)

# 预测数据
pred = linear_svc.predict(X=X_test)

ac_score = metrics.accuracy_score(y_test,pred)
#生成测试精度
cl_report = metrics.classification_report(y_test, pred)


'''
0.93006993007
             precision    recall  f1-score   support

          0       1.00      0.81      0.90        53
          1       0.90      1.00      0.95        90

avg / total       0.94      0.93      0.93       143

'''
#生成交叉验证的报告
print(ac_score)
#显示数据精度
print(cl_report)
#显示交叉验证数据集报告


# [1 0 1 1 1 0 1 0 1 1 1 1 0 0 1 1 0 1 1 1 0 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1
#  1 0 1 0 1 0 1 1 1 0 1 0 1 1 1 1 0 1 1 1 0 0 0 1 0 1 0 1 1 0 1 1 1 1 1 1 1
#  1 0 0 0 0 1 1 1 1 1 0 1 0 1 1 1 1 0 1 1 1 0 0 1 0 1 0 0 0 1 1 0 1 1 0 1 0
#  1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1]
# print(pred)


















