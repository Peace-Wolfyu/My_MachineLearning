# -*- coding: utf-8 -*-
# @Time  : 2019/12/11 19:23
# @Author : Mr.Lin


'''

多分类问题的所有指标基本上都来自于二分类指标，但是要对所有类别进行平均。多分类的
精度被定义为正确分类的样本所占的比例。同样，如果类别是不平衡的，精度并不是很
好的评估度量。想象一个三分类问题，其中 85% 的数据点属于类别 A，10% 属于类别 B，
5% 属于类别 C。在这个数据集上 85% 的精度说明了什么？一般来说，多分类结果比二分
类结果更加难以理解。除了精度，常用的工具有混淆矩阵和分类报告，我们在上一节二分
类的例子中都见过。下面我们将这两种详细的评估方法应用于对 digits 数据集中 10 种不
同的手写数字进行分类的任务：
'''
import mglearn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
digits.data, digits.target, random_state=0)
lr = LogisticRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
'''
Accuracy: 0.953
Confusion matrix:
[[37  0  0  0  0  0  0  0  0  0]
 [ 0 39  0  0  0  0  2  0  2  0]
 [ 0  0 41  3  0  0  0  0  0  0] 
 [ 0  0  1 43  0  0  0  0  0  1]
 [ 0  0  0  0 38  0  0  0  0  0]
 [ 0  1  0  0  0 47  0  0  0  0]
 [ 0  0  0  0  0  0 52  0  0  0]
 [ 0  1  0  1  1  0  0 45  0  0]
 [ 0  3  1  0  0  0  0  0 43  1]
 [ 0  0  0  1  0  1  0  0  1 44]]
'''
# print("Accuracy: {:.3f}".format(accuracy_score(y_test, pred)))
# print("Confusion matrix:\n{}".format(confusion_matrix(y_test, pred)))

scores_image = mglearn.tools.heatmap(
    confusion_matrix(y_test, pred), xlabel='Predicted label',
    ylabel='True label', xticklabels=digits.target_names,
    yticklabels=digits.target_names, cmap=plt.cm.gray_r, fmt="%d")
plt.title("Confusion matrix")
plt.gca().invert_yaxis()
'''
对于第一个类别（数字 0），它包含 37 个样本，所有这些样本都被划为类别 0（即类别 0
没有假反例）。我们之所以可以看出这一点，是因为混淆矩阵第一行中其他所有元素都为
0。我们还可以看到，没有其他数字被误分类为类别 0，这是因为混淆矩阵第一列中其他所
有元素都为 0（即类别 0 没有假正例）。但是有些数字与其他数字混在一起——比如数字 2
（第 3 行），其中有 3 个被划分到数字 3 中（第 4 列）。还有一个数字 3 被划分到数字 2 中
（第 4 行第 3 列），一个数字 8 被划分到数字 2 中（第 9 行第 3 列）。
'''
# plt.show()


'''
利用 classification_report 函数，我们可以计算每个类别的准确率、召回率和 f- 分数：
'''
'''
    precision    recall  f1-score   support

           0       1.00      1.00      1.00        37
           1       0.89      0.91      0.90        43
           2       0.95      0.93      0.94        44
           3       0.90      0.96      0.92        45
           4       0.97      1.00      0.99        38
           5       0.98      0.98      0.98        48
           6       0.96      1.00      0.98        52
           7       1.00      0.94      0.97        48
           8       0.93      0.90      0.91        48
           9       0.96      0.94      0.95        47

    accuracy                           0.95       450
   macro avg       0.95      0.95      0.95       450
weighted avg       0.95      0.95      0.95       450

不出所料，类别 0 的准确率和召回率都是完美的 1，因为这个类别中没有混淆。另一方面，
对于类别 7，准确率为 1，这是因为没有其他类别被误分类为 7；而类别 6 没有假反例，所
以召回率等于 1。我们还可以看到，模型对类别 8 和类别 3 的表现特别不好。
对于多分类问题中的不平衡数据集，最常用的指标就是多分类版本的 f- 分数。多分类 f- 分
数背后的想法是，对每个类别计算一个二分类 f- 分数，其中该类别是正类，其他所有类别
组成反类。然后，使用以下策略之一对这些按类别 f- 分数进行平均。
'''
# print(classification_report(y_test, pred))



















































