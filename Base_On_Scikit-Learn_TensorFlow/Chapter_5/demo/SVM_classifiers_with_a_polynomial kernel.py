# -*- coding: utf-8 -*-
# @Time  : 2020/1/1 20:23
# @Author : Mr.Lin
"""

添加多项式特征实现起来非常简单，并且对所有的机器学习算法
（不只是SVM）都非常有效。但问题是，如果多项式太低阶，处理
不了非常复杂的数据集，而高阶则会创造出大量的特征，导致模型变
得太慢。
幸运的是，使用SVM时，有一个魔术般的数学技巧可以应用，
这就是核技巧（稍后解释）。它产生的结果就跟添加了许多多项式特
征，甚至是非常高阶的多项式特征一样，但实际上并不需要真的添
加。因为实际没有添加任何特征，所以也就不存在数量爆炸的组合特
征了。这个技巧由SVC类来实现

这段代码使用了一个3阶多项式内核训练SVM分类器。如
左图所示。而右图是另一个使用了10阶多项式核的SVM分类器。显
然，如果模型过度拟合，你应该降低多项式阶数；反过来，如果拟合
不足，则可以尝试使之提升。超参数coef0控制的是模型受高阶多项
式还是低阶多项式影响的程度。


"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from Chapter_5.demo.Linear_SVM_classifier_using_polynomial_features import plot_predictions
from Chapter_5.demo.make_moons import X, y, plot_dataset

poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
poly_kernel_svm_clf.fit(X, y)

poly100_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))
    ])
poly100_kernel_svm_clf.fit(X, y)

plt.figure(figsize=(11, 4))

plt.subplot(121)
plot_predictions(poly_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=3, r=1, C=5$", fontsize=18)

plt.subplot(122)
plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=10, r=100, C=5$", fontsize=18)
plt.show()