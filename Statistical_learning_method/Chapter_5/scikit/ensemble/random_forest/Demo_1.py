# -*- coding: utf-8 -*-
# @Time  : 2019/12/9 20:34
# @Author : Mr.Lin


"""

由 5 棵树组成的随机森林应用到前面研究过的 two_moons 数据集上：
"""
import mglearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
random_state=42)
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)

'''
作为随机森林的一部分，树被保存在 estimator_ 属性中。我们将每棵树学到的决策边界可
视化，也将它们的总预测（即整个森林做出的预测）可视化
'''

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],
alpha=.4)
axes[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# plt.show()

'''
再举一个例子，我们将包含 100 棵树的随机森林应用在乳腺癌数据集上：
'''

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

# 在没有调节任何参数的情况下，随机森林的精度为 97%，比线性模型或单棵决策树都要
# 好。我们可以调节 max_features 参数，或者像单棵决策树那样进行预剪枝。但是，随机森
# 林的默认参数通常就已经可以给出很好的结果。


















































