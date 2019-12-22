# -*- coding: utf-8 -*-
# @Time  : 2019/12/22 14:38
# @Author : Mr.Lin

'''

训练一个二元分类器

'''
from sklearn.base import BaseEstimator
from sklearn.cross_validation import cross_val_score
import numpy as np
from Chapter_3.Demo_3.create_test_data import y_train, y_test, X_train, X
from sklearn.linear_model import SGDClassifier


y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

# 随机梯度下降（SGD）分类器进行训练
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# 拿来测试用 图片中数字为 5
some_digit = X[36000]
print(sgd_clf.predict([some_digit]))

# 用cross_val_score（）函数来评估SGDClassifier模型，采用K-fold交叉验证法，3个折叠。
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
# [ 0.96635  0.9664   0.96615]

# 一个蠢笨的分类器,它将每张图都分类成“非5”：
class Never5Classifier(BaseEstimator):

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


# 计算模型的准确度

never_5_clf = Never5Classifier()
print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
#  [ 0.9117   0.9058   0.91145]