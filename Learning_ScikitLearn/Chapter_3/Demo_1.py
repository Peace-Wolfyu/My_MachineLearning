# -*- coding: utf-8 -*-

# @Time  : 2019/12/15 14:32

# @Author : Mr.Lin

'''

3.1. 交叉验证：评估估算器的表现

'''

'''
学习预测函数的参数，并在相同数据集上进行测试是一种错误的做法: 一个仅给出测试用例标签的模型将会获得极高的分数，但对于尚未出现过的数据它则无法预测出任何有用的信息。 这种情况称为 overfitting（过拟合）. 为了避免这种情况，在进行（监督）机器学习实验时，通常取出部分可利用数据作为 test set（测试数据集） X_test, y_test。需要强调的是这里说的 “experiment(实验)” 并不仅限于学术（academic），因为即使是在商业场景下机器学习也往往是从实验开始的。下面是模型训练中典型的交叉验证工作流流程图。通过网格搜索可以确定最佳参数。
'''

'''
利用 scikit-learn 包中的 train_test_split 辅助函数可以很快地将实验数据集划分为任何训练集（training sets）和测试集（test sets）。 下面让我们载入 iris 数据集，并在此数据集上训练出线性支持向量机:
'''


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()


'''
(150, 4) (150,)

'''
print(iris.data.shape, iris.target.shape)

'''
我们能快速采样到原数据集的 40% 作为测试集，从而测试（评估）我们的分类器
'''


X_train, X_test, y_train, y_test = train_test_split(
iris.data, iris.target, test_size=0.4, random_state=0)

print("")
'''
(90, 4) (90,)

'''
print(X_train.shape, y_train.shape)
print("")
'''
(60, 4) (60,)

'''
print(X_test.shape, y_test.shape)

print("")
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
'''
0.9666666666666667

'''
print(clf.score(X_test, y_test))


'''
交叉验证（CV ） 来解决。 交叉验证仍需要测试集做最后的模型评估，但不再需要验证集。

最基本的方法被称之为，k-折交叉验证 。 k-折交叉验证将训练集划分为 k 个较小的集合（其他方法会在下面描述，主要原则基本相同）。 每一个 k 折都会遵循下面的过程：

将 k-1 份训练集子集作为 training data （训练集）训练模型，
将剩余的 1 份训练集子集用于模型验证（也就是把它当做一个测试集来计算模型的性能指标，例如准确率）。
k-折交叉验证得出的性能指标是循环计算中每个值的平均值。 该方法虽然计算代价很高，但是它不会浪费太多的数据（如固定任意测试集的情况一样）， 在处理样本数据集较少的问题（例如，逆向推理）时比较有优势。
'''

from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
# [0.96666667 1.         0.96666667 0.96666667 1.        ]
print(scores)

'''
评分估计的平均得分和 95% 置信区间由此给出
'''
#   Accuracy: 0.98 (+/- 0.03)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

'''
默认情况下，每个 CV 迭代计算的分数是估计器的 score 方法。可以通过使用 scoring 参数来改变计算方式如下:
'''

from sklearn import metrics
scores = cross_val_score(
clf, iris.data, iris.target, cv=5, scoring='f1_macro')
#   [0.96658312 1.         0.96658312 0.96658312 1.        ]
print(scores)

'''
当 cv 参数是一个整数时， cross_val_score 默认使用 KFold 或 StratifiedKFold 策略，后者会在估计器派生自 ClassifierMixin 时使用。

也可以通过传入一个交叉验证迭代器来使用其他交叉验证策略，比如:
'''


from sklearn.model_selection import ShuffleSplit
n_samples = iris.data.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
#   [0.97777778 0.97777778 1.         0.95555556 1.        ]
print(cross_val_score(clf, iris.data, iris.target, cv=cv))
print("")
print("")

'''
保留数据的数据转换

正如在训练集中保留的数据上测试一个 predictor （预测器）是很重要的一样，预处理（如标准化，特征选择等）和类似的 data transformations 也应该从训练集中学习，并应用于预测数据以进行预测:
'''

from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(
iris.data, iris.target, test_size=0.4, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
#   0.9333333333333333
print(clf.score(X_test_transformed, y_test))

'''
Pipeline 可以更容易地组合估计器，在交叉验证下使用如下:


'''
print("")
print("")
from sklearn.pipeline import make_pipeline
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
#   [0.97777778 0.93333333 0.95555556 0.93333333 0.97777778]
print(cross_val_score(clf, iris.data, iris.target, cv=cv))


'''
cross_validate 函数和多度量评估
'''

'''
cross_validate 函数与 cross_val_score 在下面的两个方面有些不同 -

它允许指定多个指标进行评估.
除了测试得分之外，它还会返回一个包含训练得分，拟合次数， score-times （得分次数）的一个字典。 It returns a dict containing training scores, fit-times and score-times in addition to the test score.
对于单个度量评估，其中 scoring 参数是一个字符串，可以调用或 None ， keys 将是 - ['test_score', 'fit_time', 'score_time']

而对于多度量评估，返回值是一个带有以下的 keys 的字典 - ['test_<scorer1_name>', 'test_<scorer2_name>', 'test_<scorer...>', 'fit_time', 'score_time']

return_train_score 默认设置为 True 。 它增加了所有 scorers(得分器) 的训练得分 keys 。如果不需要训练 scores ，则应将其明确设置为 False 。

你还可以通过设置return_estimator=True来保留在所有训练集上拟合好的估计器。

可以将多个测度指标指定为list，tuple或者是预定义评分器(predefined scorer)的名字的集合
'''


from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
scoring = ['precision_macro', 'recall_macro']
clf = svm.SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
                         cv=5)

print("")
print("")
#   ['fit_time', 'score_time', 'test_precision_macro', 'test_recall_macro', 'train_precision_macro', 'train_recall_macro']
print(sorted(scores.keys()))


print("")
print("")
#   [0.96666667 1.         0.96666667 0.96666667 1.        ]
print(scores['test_recall_macro']  )


'''
或作为一个字典 mapping 得分器名称预定义或自定义的得分函数:

'''


from sklearn.metrics.scorer import make_scorer
scoring = {'prec_macro': 'precision_macro',
'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
cv=5, return_train_score=True)
print("")
print("")
#   ['fit_time', 'score_time', 'test_prec_macro', 'test_rec_macro', 'train_prec_macro', 'train_rec_macro']
print(sorted(scores.keys()))

print("")
print("")

#   [0.975      0.975      0.99166667 0.98333333 0.98333333]
print(scores['train_rec_macro']  )

'''
这里是一个使用单一指标的 cross_validate 的例子:



'''
print("")
print("")
# scores = cross_validate(clf, iris.data, iris.target,
# scoring='precision_macro', cv=5,return_estimator=True)
# print(sorted(scores.keys()))

























































