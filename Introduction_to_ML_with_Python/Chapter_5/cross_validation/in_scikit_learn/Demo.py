# -*- coding: utf-8 -*-

# @Time  : 2019/12/11 14:29

# @Author : Mr.Lin


'''

scikit-learn 是利用 model_selection 模块中的 cross_val_score 函数来实现交叉验证的。
cross_val_score 函数的参数是我们想要评估的模型、训练数据与真实标签。我们在 iris
数据集上对 LogisticRegression 进行评估：
'''

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
iris = load_iris()



logreg = LogisticRegression()
scores = cross_val_score(logreg, iris.data, iris.target)

'''
Cross-validation scores: [ 0.96078431  0.92156863  0.95833333]

'''
# print("Cross-validation scores: {}".format(scores))

'''
默认情况下， cross_val_score 执行 3 折交叉验证，返回 3 个精度值。可以通过修改 cv 参
数来改变折数：
'''


'''
Cross-validation scores: [ 1.          0.96666667  0.93333333  0.9         1.        ]

'''
scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
# print("Cross-validation scores: {}".format(scores))


'''
总结交叉验证精度的一种常用方法是计算平均值
'''


'''
Average cross-validation score: 0.96

'''
print("Average cross-validation score: {:.2f}".format(scores.mean()))


'''
可以从交叉验证平均值中得出结论，我们预计模型的平均精度约为 96%。观察 5 折交
叉验证得到的所有 5 个精度值，我们还可以发现，折与折之间的精度有较大的变化，范围
为从 100% 精度到 90% 精度。这可能意味着模型强烈依赖于将某个折用于训练，但也可能
只是因为数据集的数据量太小。
'''











