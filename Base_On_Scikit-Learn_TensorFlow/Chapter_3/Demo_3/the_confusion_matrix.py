# -*- coding: utf-8 -*-
# @Time  : 2019/12/22 15:01
# @Author : Mr.Lin


'''


'''

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from Chapter_3.Demo_3.create_test_data import y_train, y_test, X_train

y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

# 随机梯度下降（SGD）分类器进行训练
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# 与cross_val_score（）函数一样，cross_val_predict（）函数同样执行K-fold交叉验证，但返回的不是评估分数，而是每个折叠的预测。
# 这意味着对于每个实例都可以得到一个干净的预测（“干净”的意思是模型预测时使用的数据，在其训练期间从未见过）。
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

print(confusion_matrix(y_train_5, y_train_pred))