# -*- coding: utf-8 -*-

# @Time  : 2019/12/11 15:47

# @Author : Mr.Lin


'''
在尝试调参之前，
重要的是要理解参数的含义。找到一个模型的重要参数（提供最佳泛化性能的参数）的取
值是一项棘手的任务，但对于几乎所有模型和数据集来说都是必要的。由于这项任务如此
常见，所以 scikit-learn 中有一些标准方法可以帮你完成。最常用的方法就是网格搜索
它主要是指尝试我们关心的参数的所有可能组合。
'''
from IPython.core.display import display
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.datasets import load_iris

'''
简单网格搜索
我们可以实现一个简单的网格搜索，在 2 个参数上使用 for 循环，对每种参数组合分别训
练并评估一个分类器：
'''



# 简单的网格搜索实现
from sklearn.svm import SVC

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
iris.data, iris.target, random_state=0)
print("Size of training set: {} size of test set: {}".format(
X_train.shape[0], X_test.shape[0]))
best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # 对每种参数组合都训练一个SVC
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        # 在测试集上评估SVC
        score = svm.score(X_test, y_test)
        # 如果我们得到了更高的分数，则保存该分数和对应的参数
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

'''
Size of training set: 112 size of test set: 38
Best score: 0.97
Best parameters: {'C': 100, 'gamma': 0.001}
'''
# print("Best score: {:.2f}".format(best_score))
# print("Best parameters: {}".format(best_parameters))




'''
参数过拟合的风险与验证集
看到这个结果，我们可能忍不住要报告，我们找到了一个在数据集上精度达到 97% 的模
型。然而，这种说法可能过于乐观了（或者就是错的），其原因如下：我们尝试了许多不
同的参数，并选择了在测试集上精度最高的那个，但这个精度不一定能推广到新数据上。
由于我们使用测试数据进行调参，所以不能再用它来评估模型的好坏。我们最开始需要将
数据划分为训练集和测试集也是因为这个原因。我们需要一个独立的数据集来进行评估，
一个在创建模型时没有用到的数据集。
为了解决这个问题，一种方法是再次划分数据，这样我们得到 3 个数据集：用于构建模型
的训练集，用于选择模型参数的验证集（开发集），用于评估所选参数性能的测试集
'''
print("")

'''
对数据进行 3 折划分，分为训练集、验证集和测试集
利用验证集选定最佳参数之后，我们可以利用找到的参数设置重新构建一个模型，但是要
同时在训练数据和验证数据上进行训练。这样我们可以利用尽可能多的数据来构建模型
'''


from sklearn.svm import SVC
# 将数据划分为训练+验证集与测试集
X_trainval, X_test, y_trainval, y_test = train_test_split(
iris.data, iris.target, random_state=0)
# 将训练+验证集划分为训练集与验证集
X_train, X_valid, y_train, y_valid = train_test_split(
X_trainval, y_trainval, random_state=1)
# print("Size of training set: {} size of validation set: {} size of test set:"
# " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
best_score = 0



for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # 对每种参数组合都训练一个SVC
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        # 在验证集上评估SVC
        score = svm.score(X_valid, y_valid)
        # 如果我们得到了更高的分数，则保存该分数和对应的参数
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}
# 在训练+验证集上重新构建一个模型，并在测试集上进行评估
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)

'''
Size of training set: 84 size of validation set: 28 size of test set: 38

Best score on validation set: 0.96
Best parameters:  {'C': 10, 'gamma': 0.001}
Test set score with best parameters: 0.92
'''


'''
验证集上的最高分数是 96%，这比之前略低，可能是因为我们使用了更少的数据来训练模
型（现在 X_train 更小，因为我们对数据集做了两次划分）。但测试集上的分数（这个分数
实际反映了模型的泛化能力）更低，为 92%。因此，我们只能声称对 92% 的新数据正确
分类，而不是我们之前认为的 97% ！
训练集、验证集和测试集之间的区别对于在实践中应用机器学习方法至关重要。任何根据
测试集精度所做的选择都会将测试集的信息“泄漏”（leak）到模型中。因此，保留一个单
独的测试集是很重要的，它仅用于最终评估。好的做法是利用训练集和验证集的组合完成
所有的探索性分析与模型选择，并保留测试集用于最终评估——即使对于探索性可视化也
是如此。严格来说，在测试集上对不止一个模型进行评估并选择更好的那个，将会导致对
模型精度过于乐观的估计。

'''
# print("Best score on validation set: {:.2f}".format(best_score))
# print("Best parameters: ", best_parameters)
# print("Test set score with best parameters: {:.2f}".format(test_score))

print("")
print("")
print("")


'''
带交叉验证的网格搜索
虽然将数据划分为训练集、验证集和测试集的方法（如上所述）是可行的，也相对常
用，但这种方法对数据的划分方法相当敏感。从上面代码片段的输出中可以看出，网格
搜索选择 'C': 10, 'gamma': 0.001 作为最佳参数，而 5.2.2 节的代码输出选择 'C': 100,
'gamma': 0.001 作为最佳参数。为了得到对泛化性能的更好估计，我们可以使用交叉验证
来评估每种参数组合的性能，而不是仅将数据单次划分为训练集与验证集。这种方法用代
码表示如下：
'''

import numpy as np
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # 对于每种参数组合都训练一个SVC
        svm = SVC(gamma=gamma, C=C)
        # 执行交叉验证
        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
        # 计算交叉验证平均精度
        score = np.mean(scores)
        # 如果我们得到了更高的分数，则保存该分数和对应的参数
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}
# 在训练+验证集上重新构建一个模型
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)



'''
由 于 带 交 叉 验 证 的 网 格 搜 索 是 一 种 常 用 的 调 参 方 法， 因 此 scikit-learn 提 供 了
GridSearchCV 类，它以估计器（estimator）的形式实现了这种方法。要使用 GridSearchCV
类，你首先需要用一个字典指定要搜索的参数。然后 GridSearchCV 会执行所有必要的模
型拟合。字典的键是我们要调节的参数名称（在构建模型时给出，在这个例子中是 C 和
gamma ），字典的值是我们想要尝试的参数设置。如果 C 和 gamma 想要尝试的取值为 0.001 、
0.01 、 0.1 、 1 、 10 和 100 ，可以将其转化为下面这个字典：
'''

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
# print("Parameter grid:\n{}".format(param_grid))

'''
现在我们可以使用模型（ SVC ）、要搜索的参数网格（ param_grid ）与要使用的交叉验证策
略（比如 5 折分层交叉验证）将 GridSearchCV 类实例化：
'''

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
grid_search = GridSearchCV(SVC(), param_grid, cv=5)


'''
GridSearchCV 将使用交叉验证来代替之前用过的划分训练集和验证集方法。但是，我们仍
需要将数据划分为训练集和测试集，以避免参数过拟合：
'''

X_train, X_test, y_train, y_test = train_test_split(
iris.data, iris.target, random_state=0)


'''
我们创建的 grid_search 对象的行为就像是一个分类器，我们可以对它调用标准的 fit 、
predict 和 score 方法。 2 但我们在调用 fit 时，它会对 param_grid 指定的每种参数组合都运行交叉验证
'''

grid_search.fit(X_train, y_train)



'''
拟合 GridSearchCV 对象不仅会搜索最佳参数，还会利用得到最佳交叉验证性能的参数在
整个训练数据集上自动拟合一个新模型。因此， fit 完成的工作相当于本节开头 In[21]
的代码结果。 GridSearchCV 类提供了一个非常方便的接口，可以用 predict 和 score 方
法来访问重新训练过的模型。为了评估找到的最佳参数的泛化能力，我们可以在测试集
上调用 score ：
'''


# print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))
print("")
'''
利用交叉验证选择参数，我们实际上找到了一个在测试集上精度为 97% 的模型。重要的
是，我们没有使用测试集来选择参数。我们找到的参数保存在 best_params_ 属性中，而交
叉验证最佳精度（对于这种参数设置，不同划分的平均精度）保存在 best_score_ 中：
'''

# print("Best parameters: {}".format(grid_search.best_params_))
# print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

'''

Test set score: 0.97

Best parameters: {'C': 100, 'gamma': 0.01}
Best cross-validation score: 0.97
'''


'''
同样，注意不要将 best_score_ 与模型在测试集上调用 score 方法计算得到
的泛化性能弄混。使用 score 方法（或者对 predict 方法的输出进行评估）
采用的是在整个训练集上训练的模型。而 best_score_ 属性保存的是交叉验
证的平均精度，是在训练集上进行交叉验证得到的。

'''



'''
能够访问实际找到的模型，这有时是很有帮助的，比如查看系数或特征重要性。你可以用
best_estimator_ 属性来访问最佳参数对应的模型，它是在整个训练集上训练得到的：
'''

'''
Best estimator:
SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
'''
print("")
# print("Best estimator:\n{}".format(grid_search.best_estimator_))


'''
分析交叉验证的结果
将交叉验证的结果可视化通常有助于理解模型泛化能力对所搜索参数的依赖关系。由于运
行网格搜索的计算成本相当高，所以通常最好从相对比较稀疏且较小的网格开始搜索。然
后我们可以检查交叉验证网格搜索的结果，可能也会扩展搜索范围。网格搜索的结果可以
在 cv_results_ 属性中找到，它是一个字典，其中保存了搜索的所有内容。你可以在下面
的输出中看到，它里面包含许多细节，最好将其转换成 pandas 数据框后再查看
'''


import pandas as pd
import mglearn
'''
results 中每一行对应一种特定的参数设置。对于每种参数设置，交叉验证所有划分的结
果都被记录下来，所有划分的平均值和标准差也被记录下来。由于我们搜索的是一个二维
参数网格（ C 和 gamma ），所以最适合用热图可视化（见图 5-8）。我们首先提取平均验证分
数，然后改变分数数组的形状，使其坐标轴分别对应于 C 和 gamma ：
'''
# 转换为DataFrame（数据框）
results = pd.DataFrame(grid_search.cv_results_)
# 显示前5行
# display(results.head())

scores = np.array(results.mean_test_score).reshape(6, 6)
# 对交叉验证平均分数作图
img = mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'],
ylabel='C', yticklabels=param_grid['C'], cmap="viridis")



# print(mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'],
# ylabel='C', yticklabels=param_grid['C'], cmap="viridis"))



'''
在非网格的空间中搜索
在某些情况下，尝试所有参数的所有可能组合（正如 GridSearchCV 所做的那样）并不是
一个好主意。例如， SVC 有一个 kernel 参数，根据所选择的 kernel （内核），其他参数
也是与之相关的。如果 kernel='linear' ，那么模型是线性的，只会用到 C 参数。如果
kernel='rbf' ，则需要使用 C 和 gamma 两个参数（但用不到类似 degree 的其他参数）。在
这种情况下，搜索 C 、 gamma 和 kernel 所有可能的组合没有意义：如果 kernel='linear' ，
那么 gamma 是用不到的，尝试 gamma 的不同取值将会浪费时间。为了处理这种“条
件”（conditional）参数， GridSearchCV 的 param_grid 可以是字典组成的列表（a list of
dictionaries）。列表中的每个字典可扩展为一个独立的网格。包含内核与参数的网格搜索可
能如下所示。
'''

param_grid = [{'kernel': ['rbf'],
'C': [0.001, 0.01, 0.1, 1, 10, 100],
'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
{'kernel': ['linear'],
'C': [0.001, 0.01, 0.1, 1, 10, 100]}]
# print("List of grids:\n{}".format(param_grid))

'''
在第一个网格中， kernel 参数始终等于 'rbf' （注意 kernel 是一个长度为 1 的列表），而
C 和 gamma 都是变化的。在第二个网格中， kernel 参数始终等于 'linear' ，只有 C 是变化
的。下面我们来应用这个更加复杂的参数搜索：
'''
print("")
print("")

'''
Best parameters: {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
Best cross-validation score: 0.97
'''

'''
再次查看 cv_results_ 。正如所料，如果 kernel 等于 'linear' ，那么只有 C 是变化的
'''
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))



results = pd.DataFrame(grid_search.cv_results_)
# 我们给出的是转置后的表格，这样更适合页面显示：
display(results.T)


























