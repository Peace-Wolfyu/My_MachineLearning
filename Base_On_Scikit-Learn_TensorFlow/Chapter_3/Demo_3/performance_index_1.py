# -*- coding: utf-8 -*-
# @Time  : 2019/12/22 15:38
# @Author : Mr.Lin


'''

Scikit-Learn提供了计算多种分类器指标的函数，精度和召回率也是其一

'''

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_predict
from Chapter_3.Demo_3.create_test_data import y_train, y_test, X_train
import matplotlib.pyplot as plt

y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

# 随机梯度下降（SGD）分类器进行训练
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# 与cross_val_score（）函数一样，cross_val_predict（）函数同样执行K-fold交叉验证，但返回的不是评估分数，而是每个折叠的预测。
# 这意味着对于每个实例都可以得到一个干净的预测（“干净”的意思是模型预测时使用的数据，在其训练期间从未见过）。
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# print(precision_score(y_train_5, y_train_pred))
#
# print("")
#
# print(recall_score(y_train_5, y_train_pred))
#
# print("")
#
# print(f1_score(y_train_5, y_train_pred))

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.show()
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
plot_roc_curve(fpr, tpr)
plt.show()

