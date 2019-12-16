# -*- coding: utf-8 -*-
# @Time  : 2019/12/16 20:33
# @Author : Mr.Lin

'''


朴素贝叶斯分类器
朴素贝叶斯分类器是与上一节介绍的线性模型非常相似的一种分类器，但它的训练速度往
往更快。这种高效率所付出的代价是，朴素贝叶斯模型的泛化能力要比线性分类器（如
LogisticRegression 和 LinearSVC ）稍差。
朴素贝叶斯模型如此高效的原因在于，它通过单独查看每个特征来学习参数，并从每
个特征中收集简单的类别统计数据。 scikit-learn 中实现了三种朴素贝叶斯分类器：
GaussianNB 、 BernoulliNB 和 MultinomialNB 。


'''
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from Learning_ScikitLearn.Model.Linear_Classification.Data_Source import X_test,X_train,y_train,y_test,data_y,data_X
import matplotlib.pyplot as plt

gnb = GaussianNB()

gnb.fit(X_train,y_train)

# 预测数据
pred = gnb.predict(X=X_test)

ac_score = metrics.accuracy_score(y_test,pred)
#生成测试精度
cl_report = metrics.classification_report(y_test, pred)

'''
0.937062937063
             precision    recall  f1-score   support

          0       0.96      0.87      0.91        53
          1       0.93      0.98      0.95        90

avg / total       0.94      0.94      0.94       143

'''
# print(ac_score)
# print(cl_report)
# print(data_y)

# (143,)
# (143,)
# print(y_test.shape)
# print(pred.shape)

# print(y_test)
# plt.scatter(y_test,pred)
# plt.scatter(y_test, pred, color='y', marker='o')
# plt.scatter(y_test, y_test,color='g', marker='+')
plt.show()

















