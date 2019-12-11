# -*- coding: utf-8 -*-

# @Time  : 2019/12/11 13:46

# @Author : Mr.Lin

'''

基于模型的特征选择使用一个监督机器学习模型来判断每个特征的重要性，并且仅保留最
重要的特征。用于特征选择的监督模型不需要与用于最终监督建模的模型相同。特征选择
模型需要为每个特征提供某种重要性度量，以便用这个度量对特征进行排序。决策树和基
于决策树的模型提供了 feature_importances_ 属性，可以直接编码每个特征的重要性。线
性模型系数的绝对值也可以用于表示特征重要性。正如我们在第 2 章所见，L1 惩罚的线性
模型学到的是稀疏系数，它只用到了特征的一个很小的子集。这可以被视为模型本身的一
种特征选择形式，但也可以用作另一个模型选择特征的预处理步骤。与单变量选择不同，
基于模型的选择同时考虑所有特征，因此可以获取交互项（如果模型能够获取它们的话）。
要想使用基于模型的特征选择，我们需要使用 SelectFromModel 变换器：
'''




from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

cancer = load_breast_cancer()
# 获得确定性的随机数
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
# 向数据中添加噪声特征
# 前30个特征来自数据集，后50个是噪声
X_w_noise = np.hstack([cancer.data, noise])
X_train, X_test, y_train, y_test = train_test_split(
X_w_noise, cancer.target, random_state=0, test_size=.5)

select = SelectFromModel(
RandomForestClassifier(n_estimators=100, random_state=42),
threshold="median")

'''
SelectFromModel 类选出重要性度量（由监督模型提供）大于给定阈值的所有特征。为了得
到可以与单变量特征选择进行对比的结果，我们使用中位数作为阈值，这样就可以选择一
半特征。我们用包含 100 棵树的随机森林分类器来计算特征重要性。这是一个相当复杂的
模型，也比单变量测试要强大得多。下面我们来实际拟合模型：
'''

select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
'''
X_train.shape: (284, 80)
X_train_l1.shape: (284, 40)
'''
print("X_train.shape: {}".format(X_train.shape))
print("X_train_l1.shape: {}".format(X_train_l1.shape))
print("")

mask = select.get_support()
# 将遮罩可视化——黑色为True，白色为False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
'''
这次，除了两个原始特征，其他原始特征都被选中。由于我们指定选择 40 个特征，所以
也选择了一些噪声特征。我们来看一下其性能
'''

print("")

X_test_l1 = select.transform(X_test)
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
print("Test score: {:.3f}".format(score))





















