# -*- coding: utf-8 -*-

# @Time  : 2019/12/11 14:34

# @Author : Mr.Lin




'''
分层k折交叉验证和其他策略
将数据集划分为 k 折时，从数据的前 k 分之一开始划分（正如上一节所述），这可能并不总
是一个好主意。例如，我们来看一下 iris 数据集：

'''


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()

'''
Iris labels:
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
 
 如你所见，数据的前三分之一是类别 0，中间三分之一是类别 1，最后三分之一是类别 2。
想象一下在这个数据集上进行 3 折交叉验证。第 1 折将只包含类别 0，所以在数据的第一
次划分中，测试集将只包含类别 0，而训练集只包含类别 1 和 2。由于在 3 次划分中训练
集和测试集中的类别都不相同，因此这个数据集上的 3 折交叉验证精度为 0。这没什么帮
助，因为我们在 iris 上可以得到比 0% 好得多的精度。
'''
print("Iris labels:\n{}".format(iris.target))


'''
由于简单的 k 折策略在这里失效了，所以 scikit-learn 在分类问题中不使用这种策略，而
是使用分层 k 折交叉验证（stratified k-fold cross-validation）。在分层交叉验证中，我们划分
数据，使每个折中类别之间的比例与整个数据集中的比例相同
'''



'''
举个例子，如果 90% 的样本属于类别 A 而 10% 的样本属于类别 B，那么分层交叉验证可
以确保，在每个折中 90% 的样本属于类别 A 而 10% 的样本属于类别 B。
使用分层 k 折交叉验证而不是 k 折交叉验证来评估一个分类器，这通常是一个好主意，因
为它可以对泛化性能做出更可靠的估计。在只有 10% 的样本属于类别 B 的情况下，如果
使用标准 k 折交叉验证，很可能某个折中只包含类别 A 的样本。利用这个折作为测试集的
话，无法给出分类器整体性能的信息。
对于回归问题， scikit-learn 默认使用标准 k 折交叉验证。也可以尝试让每个折表示回归
目标的不同取值，但这并不是一种常用的策略，也会让大多数用户感到意外。
'''



'''
对交叉验证的更多控制
'''



'''
利用 cv 参数来调节 cross_val_score 所使用的折数。但 scikit-learn
允许提供一个交叉验证分离器（cross-validation splitter）作为 cv 参数，来对数据划分过程
进行更精细的控制。对于大多数使用场景而言，回归问题默认的 k 折交叉验证与分类问题
的分层 k 折交叉验证的表现都很好，但有些情况下你可能希望使用不同的策略。比如说，
我们想要在一个分类数据集上使用标准 k 折交叉验证来重现别人的结果。为了实现这一
点，我们首先必须从 model_selection 模块中导入 KFold 分离器类，并用我们想要使用的
折数来将其实例化：
'''


from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=5)

logreg = LogisticRegression()

'''
然后我们可以将 kfold 分离器对象作为 cv 参数传入 cross_val_score ：

Cross-validation scores:
[ 1.          0.93333333  0.43333333  0.96666667  0.43333333]
'''
print("Cross-validation scores:\n{}".format(
cross_val_score(logreg, iris.data, iris.target, cv=kfold)))


print("")

kfold = KFold(n_splits=3)

'''
Cross-validation scores:
[ 0.  0.  0.]
'''
print("Cross-validation scores:\n{}".format(
cross_val_score(logreg, iris.data, iris.target, cv=kfold)))

'''
请记住，在 iris 数据集中每个折对应一个类别，因此学不到任何内容。解决这个问题的
另一种方法是将数据打乱来代替分层，以打乱样本按标签的排序。可以通过将 KFold 的
shuffle 参数设为 True 来实现这一点。如果我们将数据打乱，那么还需要固定 random_
state 以获得可重复的打乱结果。否则，每次运行 cross_val_score 将会得到不同的结果，
因为每次使用的是不同的划分（这可能并不是一个问题，但可能会出人意料）。在划分数
据之前将其打乱可以得到更好的结果：
'''

print("")
print("")

kfold = KFold(n_splits=3, shuffle=True, random_state=0)
'''

Cross-validation scores:
[ 0.9   0.96  0.96]
'''
print("Cross-validation scores:\n{}".format(
cross_val_score(logreg, iris.data, iris.target, cv=kfold)))




'''
留一法交叉验证
另一种常用的交叉验证方法是留一法（leave-one-out）。你可以将留一法交叉验证看作是每
折只包含单个样本的 k 折交叉验证。对于每次划分，你选择单个数据点作为测试集。这种
方法可能非常耗时，特别是对于大型数据集来说，但在小型数据集上有时可以给出更好的
估计结果：
'''
print("")
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)

'''
Number of cv iterations:  150
Mean accuracy: 0.95
'''
print("Number of cv iterations: ", len(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))
print("")




'''
打乱划分交叉验证
另一种非常灵活的交叉验证策略是打乱划分交叉验证（shuffle-split cross-validation）。在打
乱划分交叉验证中，每次划分为训练集取样 train_size 个点，为测试集取样 test_size 个
（不相交的）点。将这一划分方法重复 n_iter 次。：

'''

'''
下面的代码将数据集划分为 50% 的训练集和 50% 的测试集，共运行 10 次迭代
1 ：
'''


print("")
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
'''
Cross-validation scores:
[ 0.90666667  0.92        0.96        0.94666667  0.97333333  0.97333333
  0.97333333  0.92        0.97333333  0.8       ]
'''
print("Cross-validation scores:\n{}".format(scores))


'''
打乱划分交叉验证可以在训练集和测试集大小之外独立控制迭代次数，这有时是很有帮助
的。它还允许在每次迭代中仅使用部分数据，这可以通过设置 train_size 与 test_size 之
和不等于 1 来实现。用这种方法对数据进行二次采样可能对大型数据上的试验很有用。
ShuffleSplit 还有一种分层的形式，其名称为 StratifiedShuffleSplit ，它可以为分类任
务提供更可靠的结果
'''






















































































































