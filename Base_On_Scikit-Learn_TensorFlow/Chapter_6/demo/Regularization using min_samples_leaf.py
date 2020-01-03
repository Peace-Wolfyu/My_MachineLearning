# -*- coding: utf-8 -*-
# @Time  : 2020/1/3 15:15
# @Author : Mr.Lin
"""
决策树极少对训练数据做出假设（比如线性模型就正好相反，它
显然假设数据是线性的）。如果不加以限制，树的结构将跟随训练集
变化，严密拟合，并且很可能过度拟合。这种模型通常被称为非参数
模型，这不是说它不包含任何参数（事实上它通常有很多参数），而
是指在训练之前没有确定参数的数量，导致模型结构自由而紧密地贴
近数据。相应的参数模型，比如线性模型，则有预先设定好的一部分
参数，因此其自由度受限，从而降低了过度拟合的风险（但是增加了
拟合不足的风险）。
为避免过度拟合，需要在训练过程中降低决策树的自由度。现在
你应该知道，这个过程被称为正则化。正则化超参数的选择取决于你
所使用的模型，但是通常来说，至少可以限制决策树的最大深度。在
Scikit-Learn中，这由超参数max_depth控制（默认值为None，意味着
无限制）。减小max_depth可使模型正则化，从而降低过度拟合的风
险。
DecisionTreeClassifier类还有一些其他的参数，同样可以限制决
策树的形状：min_samples_split（分裂前节点必须有的最小样本
数），min_samples_leaf（叶节点必须有的最小样本数量），
min_weight_fraction_leaf（跟min_samples_leaf一样，但表现为加权实
例总数的占比），max_leaf_nodes（最大叶节点数量），以及
max_features（分裂每个节点评估的最大特征数量）。增大超参数
min_*或是减小max_*将使模型正则化。

还可以先不加约束地训练模型，然后再对不必要的节点进行
剪枝（删除）。如果一个节点的子节点全部为叶节点，则该节点可被
认为不必要，除非它所表示的纯度提升有重要的统计意义。标准统计
测试，比如χ 2 测试，是用来估算“提升纯粹是出于偶然”（被称为虚假
设）的概率。如果这个概率（称之为p值）高于一个给定阈值（通常
是5%，由超参数控制），那么这个节点可被认为不必要，其子节点
可被删除。直到所有不必要的节点都被删除，剪枝过程结束。



左图使用默认参数（即无约束）来训练决策树，右图的决策树应
用min_samples_leaf=4进行训练。很明显，左图模型过度拟合，右图
的泛化效果更佳。
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from Chapter_6.demo.d import plot_decision_boundary

Xm, ym = make_moons(n_samples=100, noise=0.25, random_state=53)

deep_tree_clf1 = DecisionTreeClassifier(random_state=42)
deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
deep_tree_clf1.fit(Xm, ym)
deep_tree_clf2.fit(Xm, ym)

plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_decision_boundary(deep_tree_clf1, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("No restrictions", fontsize=16)
plt.subplot(122)
plot_decision_boundary(deep_tree_clf2, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)
plt.show()