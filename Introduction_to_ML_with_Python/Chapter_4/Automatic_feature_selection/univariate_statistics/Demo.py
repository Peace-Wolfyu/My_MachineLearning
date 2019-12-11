# -*- coding: utf-8 -*-

# @Time  : 2019/12/11 13:40

# @Author : Mr.Lin

'''
在单变量统计中，我们计算每个特征和目标值之间的关系是否存在统计显著性，然后选
择具有最高置信度的特征。对于分类问题，这也被称为方差分析（analysis of variance，
ANOVA）。这些测试的一个关键性质就是它们是单变量的（univariate），即它们只单独考
虑每个特征。因此，如果一个特征只有在与另一个特征合并时才具有信息量，那么这个特
征将被舍弃。单变量测试的计算速度通常很快，并且不需要构建模型。另一方面，它们完
全独立于你可能想要在特征选择之后应用的模型。
'''

'''
想要在 scikit-learn 中使用单变量特征选择，你需要选择一项测试——对分类问题通常
是 f_classif （默认值），对回归问题通常是 f_regression ——然后基于测试中确定的 p 值
来选择一种舍弃特征的方法。所有舍弃参数的方法都使用阈值来舍弃所有 p 值过大的特征
（意味着它们不可能与目标值相关）。计算阈值的方法各有不同，最简单的是 SelectKBest
和 SelectPercentile ，前者选择固定数量的 k 个特征，后者选择固定百分比的特征。我们
将分类的特征选择应用于 cancer 数据集。为了使任务更难一点，我们将向数据中添加一些
没有信息量的噪声特征。我们期望特征选择能能够识别没有信息量的特征并删除它们：
'''


from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


cancer = load_breast_cancer()
# 获得确定性的随机数
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
# 向数据中添加噪声特征
# 前30个特征来自数据集，后50个是噪声
X_w_noise = np.hstack([cancer.data, noise])
X_train, X_test, y_train, y_test = train_test_split(
X_w_noise, cancer.target, random_state=0, test_size=.5)
# 使用f_classif（默认值）和SelectPercentile来选择50%的特征
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
# 对训练集进行变换
X_train_selected = select.transform(X_train)
'''
X_train.shape: (284, 80)
X_train_selected.shape: (284, 40)

如你所见，特征的数量从 80 减少到 40（原始特征数量的 50%）。我们可以用 get_
support 方法来查看哪些特征被选中，它会返回所选特征的布尔遮罩（mask）（其可视化


'''
print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))



mask = select.get_support()
print(mask)
# 将遮罩可视化——黑色为True，白色为False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
'''
[ True  True  True  True  True  True  True  True  True False  True False
  True  True  True  True  True  True False False  True  True  True  True
  True  True  True  True  True  True False False False  True False  True
 False False  True False False False False  True False False  True False
 False  True False  True False False False False False False  True False
  True False False False False  True False  True False False False False
  True  True False  True False False False False]
'''
# plt.show()

'''
你可以从遮罩的可视化中看出，大多数所选择的特征都是原始特征，并且大多数噪声特征
都已被删除。但原始特征的还原并不完美。我们来比较 Logistic 回归在所有特征上的性能
与仅使用所选特征的性能：
'''



print("")
from sklearn.linear_model import LogisticRegression
# 对测试数据进行变换

'''
Score with all features: 0.930
Score with only selected features: 0.940


在这个例子中，删除噪声特征可以提高性能，即使丢失了某些原始特征。这是一个非常简
单的假想示例，在真实数据上的结果要更加复杂。不过，如果特征量太大以至于无法构建
模型，或者你怀疑许多特征完全没有信息量，那么单变量特征选择还是非常有用的。
'''
X_test_selected = select.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Score with all features: {:.3f}".format(lr.score(X_test, y_test)))
lr.fit(X_train_selected, y_train)
print("Score with only selected features: {:.3f}".format(
lr.score(X_test_selected, y_test)))








