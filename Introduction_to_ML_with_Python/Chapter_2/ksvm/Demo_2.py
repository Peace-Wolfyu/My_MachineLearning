# -*- coding: utf-8 -*-
# @Time  : 2019/12/10 18:59
# @Author : Mr.Lin


'''

向数据表示中添加非线性特征，可以让线性模型变得更强大。但是，
通常来说我们并不知道要添加哪些特征，而且添加许多特征（比如 100 维特征空间所有可
能的交互项）的计算开销可能会很大。幸运的是，有一种巧妙的数学技巧，让我们可以在
更高维空间中学习分类器，而不用实际计算可能非常大的新的数据表示。这种技巧叫作核
技巧（kernel trick），它的原理是直接计算扩展特征表示中数据点之间的距离（更准确地说
是内积），而不用实际对扩展进行计算。
'''
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

'''
对于支持向量机，将数据映射到更高维空间中有两种常用的方法：一种是多项式核，在一
定阶数内计算原始特征所有可能的多项式（比如 feature1 ** 2 * feature2 ** 5 ）；另一
种是径向基函数（radial basis function，RBF）核，也叫高斯核。高斯核有点难以解释，因
为它对应无限维的特征空间。一种对高斯核的解释是它考虑所有阶数的所有可能的多项
式，但阶数越高，特征的重要性越小
'''

'''
理解SVM
'''


'''
在训练过程中，SVM 学习每个训练数据点对于表示两个类别之间的决策边界的重要性。通
常只有一部分训练数据点对于定义决策边界来说很重要：位于类别之间边界上的那些点。
这些点叫作支持向量（support vector），支持向量机正是由此得名。
'''


'''
想要对新样本点进行预测，需要测量它与每个支持向量之间的距离。分类决策是基于它与
支持向量之间的距离以及在训练过程中学到的支持向量重要性（保存在 SVC 的 dual_coef_
属性中）来做出的。
'''


'''
下列代码将在 forge 数据集上训练 SVM 并创建此图
'''

from sklearn.svm import SVC
import mglearn


X,y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf',C=10,gamma=0.1).fit(X=X,y=y)
import matplotlib.pyplot as plt

mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# 画出支持向量
sv = svm.support_vectors_
# 支持向量的类别标签由dual_coef_的正负号给出

sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
# 决策边界用黑色表示，支持向
# 量是尺寸较大的点

'''
RBF 核 SVM 给出的决策边界和支持向量

在这个例子中，SVM 给出了非常平滑且非线性（不是直线）的边界。这里我们调节了两
个参数： C 参数和 gamma 参数
'''
# plt.show()

'''
SVM调参
'''

'''
gamma 参数是上一节给出的公式中的参数，用于控制高斯核的宽度。它决定了点与点之间
“靠近”是指多大的距离。 C 参数是正则化参数，与线性模型中用到的类似。它限制每个点
的重要性（或者更确切地说，每个点的 dual_coef_ ）。
'''

'''
来看一下，改变这些参数时会发生什么
'''

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
axes[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"],
ncol=4, loc=(.9, 1.2))

'''
从左到右，我们将参数 gamma 的值从 0.1 增加到 10 。 gamma 较小，说明高斯核的半径较大，
许多点都被看作比较靠近。这一点可以在图中看出：左侧的图决策边界非常平滑，越向右
的图决策边界更关注单个点。小的 gamma 值表示决策边界变化很慢，生成的是复杂度较低
的模型，而大的 gamma 值则会生成更为复杂的模型。
从上到下，我们将参数 C 的值从 0.1 增加到 1000 。与线性模型相同， C 值很小，说明模型
非常受限，每个数据点的影响范围都有限。你可以看到，左上角的图中，决策边界看起来
几乎是线性的，误分类的点对边界几乎没有任何影响。再看左下角的图，增大 C 之后这些
点对模型的影响变大，使得决策边界发生弯曲来将这些点正确分类。
'''
# plt.show()

'''

将 RBF 核 SVM 应用到乳腺癌数据集上。默认情况下， C=1 ， gamma=1/n_features ：
'''

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, random_state=0)
svc = SVC()
svc.fit(X_train, y_train)

'''
Accuracy on training set: 1.00
Accuracy on test set: 0.63

这个模型在训练集上的分数十分完美，但在测试集上的精度只有 63%，存在相当严重的过
拟合。虽然 SVM 的表现通常都很好，但它对参数的设定和数据的缩放非常敏感。特别地，
它要求所有特征有相似的变化范围。


'''
# print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
# print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))


'''
来看一下每个特征的最小值和最大值，它们绘制
在对数坐标上
'''
plt.plot(X_train.min(axis=0), 'o', label="min")
plt.plot(X_train.max(axis=0), '^', label="max")
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")

'''
从这张图中，我们可以确定乳腺癌数据集的特征具有完全不同的数量级。这对其他模型来
说（比如线性模型）可能是小问题，但对核 SVM 却有极大影响。我们来研究处理这个问
题的几种方法。
'''
# plt.show()


'''
为SVM预处理数据
'''
'''
解决这个问题的一种方法就是对每个特征进行缩放，使其大致都位于同一范围。核 SVM
常用的缩放方法就是将所有特征缩放到 0 和 1 之间
'''


# 计算训练集中每个特征的最小值
min_on_training = X_train.min(axis=0)
# 计算训练集中每个特征的范围（最大值-最小值）
range_on_training = (X_train - min_on_training).max(axis=0)
# 减去最小值，然后除以范围
# 这样每个特征都是min=0和max=1
X_train_scaled = (X_train - min_on_training) / range_on_training

'''
Minimum for each feature
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0.]
Maximum for each feature
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1.]
'''
# print("Minimum for each feature\n{}".format(X_train_scaled.min(axis=0)))
# print("Maximum for each feature\n {}".format(X_train_scaled.max(axis=0)))


# 利用训练集的最小值和范围对测试集做相同的变换
X_test_scaled = (X_test - min_on_training) / range_on_training

svc = SVC()
svc.fit(X_train_scaled, y_train)

'''
Accuracy on training set: 0.948
Accuracy on test set: 0.951
'''

'''
数据缩放的作用很大！实际上模型现在处于欠拟合的状态，因为训练集和测试集的性能非
常接近，但还没有接近 100% 的精度
'''
print("Accuracy on training set: {:.3f}".format(
svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))


'''
从这里开始，我们可以尝试增大 C 或 gamma 来拟合
更为复杂的模型。例如：
'''

svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)
'''
Accuracy on training set: 0.988
Accuracy on test set: 0.972

在这个例子中，增大 C 可以显著改进模型，得到 97.2% 的精度


'''
print("Accuracy on training set: {:.3f}".format(
svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))
