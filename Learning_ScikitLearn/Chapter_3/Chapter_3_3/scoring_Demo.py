# -*- coding: utf-8 -*-
# @Time  : 2019/12/15 20:59
# @Author : Mr.Lin

'''

scoring 参数: 定义模型评估规则

'''


from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf = svm.SVC(probability=True, random_state=0)

# [-0.0757148  -0.1682413  -0.07057969]

print(cross_val_score(clf, X, y, scoring='neg_log_loss'))

# cross_val_score(model, X, y, scoring='wrong_choice')



