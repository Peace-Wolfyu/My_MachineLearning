# -*- coding: utf-8 -*-
# @Time  : 2019/12/11 19:09
# @Author : Mr.Lin


'''
二分类指标
二分类可能是实践中最常见的机器学习应用，也是概念最简单的应用。但是，即使是评估
这个简单任务也仍有一些注意事项。在深入研究替代指标之前，我们先看一下测量精度可
能会如何误导我们。请记住，对于二分类问题，我们通常会说正类（positive class）和反类
（negative class），而正类是我们要寻找的类。

'''
from sklearn.model_selection import train_test_split

'''
错误类型
'''

'''
一种可能的错误是健康的患者被诊断为阳性，导致需要进行额外的测试。这给患者带来
了一些费用支出和不便（可能还有精神上的痛苦）。错误的阳性预测叫作假正例（false
positive）。另一种可能的错误是患病的人被诊断为阴性，因而不会接受进一步的检查和治
疗。未诊断出的癌症可能导致严重的健康问题，甚至可能致命。这种类型的错误（错误的
阴性预测）叫作假反例（false negative）。在统计学中，假正例也叫作第一类错误（type I
error），假反例也叫作第二类错误（type II error）。我们将坚持使用“假正例”和“假反例”
的说法，因为它们的含义更加明确，也更好记。在癌症诊断的例子中，显然，我们希望尽
量避免假反例，而假正例可以被看作是小麻烦。
'''

'''
不平衡数据集
'''
'''
如果在两个类别中，一个类别的出现次数比另一个多很多，那么错误类型将发挥重要作
用。这在实践中十分常见，一个很好的例子是点击（click-through）预测，其中每个数据点
表示一个“印象”（impression），即向用户展示的一个物项。这个物项可能是广告、相关的
故事，或者是在社交媒体网站上关注的相关人员。目标是预测用户是否会点击看到的某个
特定物项（表示他们感兴趣）。用户对互联网上显示的大多数内容（尤其是广告）都不会
点击。你可能需要向用户展示 100 个广告或文章，他们才会找到足够有趣的内容来点击查
看。这样就会得到一个数据集，其中每 99 个“未点击”的数据点才有 1 个“已点击”的
数据点。换句话说，99% 的样本属于“未点击”类别。这种一个类别比另一个类别出现次
数多很多的数据集，通常叫作不平衡数据集（imbalanced dataset）或者具有不平衡类别的
数据集（dataset with imbalanced classes）。在实际当中，不平衡数据才是常态，而数据中感
兴趣事件的出现次数相同或相似的情况十分罕见
'''





from sklearn.datasets import load_digits
import numpy as np
digits = load_digits()
y = digits.target == 9
X_train, X_test, y_train, y_test = train_test_split(
digits.data, y, random_state=0)


'''
使用 DummyClassifier 来始终预测多数类（这里是“非 9”），以查看精度提供的信
息量有多么少：
'''

from sklearn.dummy import DummyClassifier
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)

'''
Unique predicted labels: [False]
Test score: 0.90
'''
# print("Unique predicted labels: {}".format(np.unique(pred_most_frequent)))
# print("Test score: {:.2f}".format(dummy_majority.score(X_test, y_test)))


'''
得到了接近 90% 的精度，却没有学到任何内容。这个结果可能看起来相当好，但请
思考一会儿。想象一下，有人告诉你他们的模型精度达到 90%。你可能会认为他们做得很
好。但根据具体问题，也可能是仅预测了一个类别！我们将这个结果与使用一个真实分类
器的结果进行对比：
'''
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
'''
Test score: 0.92

'''
# print("Test score: {:.2f}".format(tree.score(X_test, y_test)))


'''
从精度来看， DecisionTreeClassifier 仅比常数预测稍好一点。这可能表示我们使用
DecisionTreeClassifier 的方法有误，也可能是因为精度实际上在这里不是一个很好的度量。
为了便于对比，我们再评估两个分类器， LogisticRegression 与默认的 DummyClassifier ，
其中后者进行随机预测，但预测类别的比例与训练集中的比例相同
5 
'''

from sklearn.linear_model import LogisticRegression
dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
# print("dummy score: {:.2f}".format(dummy.score(X_test, y_test)))
logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
# print("logreg score: {:.2f}".format(logreg.score(X_test, y_test)))

'''
dummy score: 0.83
logreg score: 0.98

显而易见，产生随机输出的虚拟分类器是所有分类器中最差的（精度最低），而
LogisticRegression 则给出了非常好的结果。但是，即使是随机分类器也得到了超过 80% 的
精度。这样很难判断哪些结果是真正有帮助的。这里的问题在于，要想对这种不平衡数据的
预测性能进行量化，精度并不是一种合适的度量。
'''

'''
混淆矩阵
对于二分类问题的评估结果，一种最全面的表示方法是使用混淆矩阵（confusion matrix）。
我们利用 confusion_matrix 函数来检查上一节中 LogisticRegression 的预测结果。我们已
经将测试集上的预测结果保存在 pred_logreg 中：
'''
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, pred_logreg)
'''
Confusion matrix:
[[401   2]
 [  8  39]]
'''
'''
confusion_matrix 的输出是一个 2×2 数组，其中行对应于真实的类别，列对应于预测的类
别。数组中每个元素给出属于该行对应类别（这里是“非 9”和“9”）的样本被分类到该
列对应类别中的数量
'''
'''
混淆矩阵主对角线
6
上的元素对应于正确的分类，而其他元素则告诉我们一个类别中有多少
样本被错误地划分到其他类别中。
'''
# print("Confusion matrix:\n{}".format(confusion))


'''
用混淆矩阵来比较前面拟合过的模型（两个虚拟模型、决策树和 Logistic 回归）：
'''

'''
Most frequent class:
[[403   0]
 [ 47   0]]

Dummy model:
[[371  32]
 [ 41   6]]

Decision tree:
[[390  13]
 [ 24  23]]

Logistic Regression
[[401   2]
 [  8  39]]
'''
'''
观察混淆矩阵，很明显可以看出 pred_most_frequent 有问题，因为它总是预测同一个类
别。另一方面， pred_dummy 的真正例数量很少（4 个），特别是与假反例和假正例的数量相
比——假正例的数量竟然比真正例还多！决策树的预测比虚拟预测更有意义，即使二者精
度几乎相同。最后，我们可以看到，Logistic 回归在各方面都比 pred_tree 要好：它的真正
例和真反例的数量更多，而假正例和假反例的数量更少。从这个对比中可以明确看出，只
有决策树和 Logistic 回归给出了合理的结果，并且 Logistic 回归的效果全面好于决策树。
但是，检查整个混淆矩阵有点麻烦，虽然我们通过观察矩阵的各个方面得到了很多深入见
解，但是这个过程是人工完成的，也是非常定性的。
'''


print("Most frequent class:")
print(confusion_matrix(y_test, pred_most_frequent))
print("\nDummy model:")
print(confusion_matrix(y_test, pred_dummy))
print("\nDecision tree:")
print(confusion_matrix(y_test, pred_tree))
print("\nLogistic Regression")
print(confusion_matrix(y_test, pred_logreg))





































