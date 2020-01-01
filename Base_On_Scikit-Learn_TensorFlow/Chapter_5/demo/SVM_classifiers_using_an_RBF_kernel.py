# -*- coding: utf-8 -*-
# @Time  : 2020/1/1 20:59
# @Author : Mr.Lin
"""

与多项式特征方法一样，相似特征法也可以用任意机器学习算
法，但是要计算出所有附加特征，其计算代价可能非常昂贵，尤其是
对大型训练集来说。然而，核技巧再一次施展了它的SVM魔术：它
能够产生的结果就跟添加了许多相似特征一样，但实际上也并不需要
添加。我们来使用SVC类试试高斯RBF核：

图的左下方显示了这个模型。其他图显示了超参数
gamma（）和C使用不同值时的模型。增加gamma值会使钟形曲线变
得更窄（左图），因此每个实例的影响范围随之变小：决策
边界变得更不规则，开始围着单个实例绕弯。反过来，减小gamma值
使钟形曲线变得更宽，因而每个实例的影响范围增大，决策边界变得
更平坦。所以就像是一个正则化的超参数：模型过度拟合，就降低它
的值，如果拟合不足则提升它的值（类似超参数C）。

"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from Chapter_5.demo.Linear_SVM_classifier_using_polynomial_features import plot_predictions
from Chapter_5.demo.make_moons import plot_dataset, X, y

rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ])
rbf_kernel_svm_clf.fit(X, y)
from sklearn.svm import SVC

gamma1, gamma2 = 0.1, 5
C1, C2 = 0.001, 1000
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

svm_clfs = []
for gamma, C in hyperparams:
    rbf_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))
        ])
    rbf_kernel_svm_clf.fit(X, y)
    svm_clfs.append(rbf_kernel_svm_clf)

plt.figure(figsize=(11, 7))

for i, svm_clf in enumerate(svm_clfs):
    plt.subplot(221 + i)
    plot_predictions(svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)
plt.show()































