# -*- coding: utf-8 -*-

# @Time  : 2019/12/11 13:56

# @Author : Mr.Lin

''''

在单变量测试中，我们没有使用模型，而在基于模型的选择中，我们使用了单个模型来选
择特征。在迭代特征选择中，将会构建一系列模型，每个模型都使用不同数量的特征。有
两种基本方法：开始时没有特征，然后逐个添加特征，直到满足某个终止条件；或者从所
有特征开始，然后逐个删除特征，直到满足某个终止条件。由于构建了一系列模型，所
以这些方法的计算成本要比前面讨论过的方法更高。其中一种特殊方法是递归特征消除
（recursive feature elimination，RFE），它从所有特征开始构建模型，并根据模型舍弃最不重
要的特征，然后使用除被舍弃特征之外的所有特征来构建一个新模型，如此继续，直到仅
剩下预设数量的特征。为了让这种方法能够运行，用于选择的模型需要提供某种确定特征
重要性的方法，正如基于模型的选择所做的那样。下面我们使用之前用过的同一个随机森
林模型
'''

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
cancer = load_breast_cancer()
# 获得确定性的随机数
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
# 向数据中添加噪声特征
# 前30个特征来自数据集，后50个是噪声
X_w_noise = np.hstack([cancer.data, noise])
X_train, X_test, y_train, y_test = train_test_split(
X_w_noise, cancer.target, random_state=0, test_size=.5)


select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),
n_features_to_select=40)

select.fit(X_train, y_train)
# 将选中的特征可视化：
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")


'''
与单变量选择和基于模型的选择相比，迭代特征选择的结果更好，但仍然漏掉了一个特
征。运行上述代码需要的时间也比基于模型的选择长得多，因为对一个随机森林模型训练
了 40 次，每运行一次删除一个特征。我们来测试一下使用 RFE 做特征选择时 Logistic 回
归模型的精度：
'''
X_train_rfe= select.transform(X_train)
X_test_rfe= select.transform(X_test)
score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)

'''
Test score: 0.951


'''
print("Test score: {:.3f}".format(score))

'''
如果你不确定何时选择使用哪些特征作为机器学习算法的输入，那么自动化特征选择可能
特别有用。它还有助于减少所需要的特征数量，加快预测速度，或允许可解释性更强的模
型。在大多数现实情况下，使用特征选择不太可能大幅提升性能，但它仍是特征工程工具
箱中一个非常有价值的工具
'''















