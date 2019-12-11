# -*- coding: utf-8 -*-
# @Time  : 2019/12/11 19:35
# @Author : Mr.Lin

'''

因此，大多数机器学习应用不仅需要应用单个算法，而且还需要将许多不同的处理步
骤和机器学习模型链接在一起。本章将介绍如何使用 Pipeline 类来简化构建变换和模型链
的过程。我们将重点介绍如何将 Pipeline 和 GridSearchCV 结合起来，从而同时搜索所有
处理步骤中的参数。
举一个例子来说明模型链的重要性。我们知道，可以通过使用 MinMaxScaler 进行预处理来
大大提高核 SVM 在 cancer 数据集上的性能。下面这些代码实现了划分数据、计算最小值
和最大值、缩放数据与训练 SVM：
'''

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# 加载并划分数据
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, random_state=0)
# 计算训练数据的最小值和最大值
scaler = MinMaxScaler().fit(X_train)

# 对训练数据进行缩放
X_train_scaled = scaler.transform(X_train)


svm = SVC()
# 在缩放后的训练数据上学习SVM
svm.fit(X_train_scaled, y_train)
# 对测试数据进行缩放，并计算缩放后的数据的分数
X_test_scaled = scaler.transform(X_test)
'''
Test score: 0.95

'''
# print("Test score: {:.2f}".format(svm.score(X_test_scaled, y_test)))

'''
现在，假设我们希望利用 GridSearchCV 找到更好的 SVC 参数，正如第 5 章中所做的那样。
我们应该怎么做？一种简单的方法可能如下所示：
'''


from sklearn.model_selection import GridSearchCV
# 只是为了便于说明，不要在实践中使用这些代码！
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
'''
Best cross-validation accuracy: 0.98
Best set score: 0.97
Best parameters:  {'C': 1, 'gamma': 1}
'''
# print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
# print("Best set score: {:.2f}".format(grid.score(X_test_scaled, y_test)))
# print("Best parameters: ", grid.best_params_)

'''
我们利用缩放后的数据对 SVC 参数进行网格搜索。但是，上面的代码中有一个不易察
觉的陷阱。在缩放数据时，我们使用了训练集中的所有数据来找到训练的方法。然后，我
们使用缩放后的训练数据来运行带交叉验证的网格搜索。对于交叉验证中的每次划分，原
始训练集的一部分被划分为训练部分，另一部分被划分为测试部分。测试部分用于度量在
训练部分上所训练的模型在新数据上的表现。但是，我们在缩放数据时已经使用过测试部
分中所包含的信息。请记住，交叉验证每次划分的测试部分都是训练集的一部分，我们使
用整个训练集的信息来找到数据的正确缩放。
对于模型来说，这些数据与新数据看起来截然不同。如果我们观察新数据（比如测试集中
的数据），那么这些数据并没有用于对训练数据进行缩放，其最大值和最小值也可能与训
练数据不同
'''

'''
因此，对于建模过程，交叉验证中的划分无法正确地反映新数据的特征。我们已经将这部
分数据的信息泄露（leak）给建模过程。这将导致在交叉验证过程中得到过于乐观的结果，
并可能会导致选择次优的参数。
为了解决这个问题，在交叉验证的过程中，应该在进行任何预处理之前完成数据集的划
分。任何从数据集中提取信息的处理过程都应该仅应用于数据集的训练部分，因此，任何
交叉验证都应该位于处理过程的“最外层循环”。
在 scikit-learn 中，要想使用 cross_val_score 函数和 GridSearchCV 函数实现这一点，可
以使用 Pipeline 类。 Pipeline 类可以将多个处理步骤合并（glue）为单个 scikit-learn 估
计器。 Pipeline 类本身具有 fit 、 predict 和 score 方法，其行为与 scikit-learn 中的其
他模型相同。 Pipeline 类最常见的用例是将预处理步骤（比如数据缩放）与一个监督模型
（比如分类器）链接在一起。
'''





from sklearn.pipeline import Pipeline
'''
这里我们创建了两个步骤：第一个叫作 "scaler" ，是 MinMaxScaler 的实例；第二个叫作
"svm" ，是 SVC 的实例。现在我们可以像任何其他 scikit-learn 估计器一样来拟合这个管道：
'''
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])

'''
这里 pipe.fit 首先对第一个步骤（缩放器）调用 fit ，然后使用该缩放器对训练数据进
行变换，最后用缩放后的数据来拟合 SVM。要想在测试数据上进行评估，我们只需调用
pipe.score ：
'''
pipe.fit(X_train, y_train)

'''
Test score: 0.95

'''
'''
如果对管道调用 score 方法，则首先使用缩放器对测试数据进行变换，然后利用缩放后
的测试数据对 SVM 调用 score 方法。如你所见，这个结果与我们从本章开头的代码得到
的结果（手动进行数据变换）是相同的。利用管道，我们减少了“预处理 + 分类”过程
所需要的代码量。但是，使用管道的主要优点在于，现在我们可以在 cross_val_score 或
GridSearchCV 中使用这个估计器
'''
# print("Test score: {:.2f}".format(pipe.score(X_test, y_test)))

'''
在网格搜索中使用管道的工作原理与使用任何其他估计器都相同。我们定义一个需要搜索
的参数网格，并利用管道和参数网格构建一个 GridSearchCV 。不过在指定参数网格时存在
一处细微的变化。我们需要为每个参数指定它在管道中所属的步骤。我们要调节的两个参
数 C 和 gamma 都是 SVC 的参数，属于第二个步骤。我们给这个步骤的名称是 "svm" 。为管
道定义参数网格的语法是为每个参数指定步骤名称，后面加上 __ （双下划线），然后是参
数名称。因此，要想搜索 SVC 的 C 参数，必须使用 "svm__C" 作为参数网格字典的键，对
gamma 参数也是同理：
'''
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

'''
有了这个参数网格，我们可以像平常一样使用 GridSearchCV 
'''

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

'''
Best cross-validation accuracy: 0.98
Test set score: 0.97
Best parameters: {'svm__C': 1, 'svm__gamma': 1}
'''
# print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
# print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
# print("Best parameters: {}".format(grid.best_params_))

'''
用 make_pipeline 方便地创建管道
利用上述语法创建管道有时有点麻烦，我们通常不需要为每一个步骤提供用户指定的名
称。有一个很方便的函数 make_pipeline ，可以为我们创建管道并根据每个步骤所属的类
为其自动命名。 make_pipeline 的语法如下所示：
'''
from sklearn.pipeline import make_pipeline
# 标准语法
pipe_long = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC(C=100))])
# 缩写语法
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))

'''
管道对象 pipe_long 和 pipe_short 的作用完全相同，但 pipe_short 的步骤是自动命名的。
我们可以通过查看 steps 属性来查看步骤的名称：
'''

'''
Pipeline steps:
[('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('svc', SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False))]
'''
'''
这两个步骤被命名为 minmaxscaler 和 svc 。一般来说，步骤名称只是类名称的小写版本。
如果多个步骤属于同一个类，则会附加一个数字：
'''
print("Pipeline steps:\n{}".format(pipe_short.steps))























































