# -*- coding: utf-8 -*-
# @Time  : 2019/12/15 20:01
# @Author : Mr.Lin


'''

留一交叉验证 (LOO)
LeaveOneOut (或 LOO) 是一个简单的交叉验证。每个学习集都是通过除了一个样本以外的所有样本创建的，测试集是被留下的样本。 因此，对于 n 个样本，我们有 n 个不同的训练集和 n 个不同的测试集。这种交叉验证程序不会浪费太多数据，因为只有一个样本是从训练集中删除掉的:

'''


from sklearn.model_selection import LeaveOneOut, cross_val_score
from Learning_ScikitLearn.Chapter_3.Data import iris,logreg

X = [1, 2, 3, 4]
loo = LeaveOneOut()
# LOO 潜在的用户选择模型应该权衡一些已知的警告。 当与 k 折交叉验证进行比较时，可以从 n 样本中构建 n 模型，而不是 k 模型，其中 n > k 。 此外，每个在 n - 1 个样本而不是在 (k-1) n / k 上进行训练。在两种方式中，假设 k 不是太大，并且 k < n ， LOO 比 k 折交叉验证计算开销更加昂贵。
#
# 就精度而言， LOO 经常导致较高的方差作为测试误差的估计器。直观地说，因为 n 个样本中的 n - 1 被用来构建每个模型，折叠构建的模型实际上是相同的，并且是从整个训练集建立的模型。
#
# 但是，如果学习曲线对于所讨论的训练大小是陡峭的，那么 5- 或 10- 折交叉验证可以泛化误差增高。
#
# 作为一般规则，大多数作者和经验证据表明， 5- 或者 10- 交叉验证应该优于 LOO 。

# [1 2 3]  [0]
# [0 2 3]  [1]
# [0 1 3]  [2]
# [0 1 2]  [3]
for train, test in loo.split(X):
    print("%s  %s" % (train, test))

scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)

# 0.9533333333333334
print(scores.mean())

















