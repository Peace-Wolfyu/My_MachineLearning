# -*- coding: utf-8 -*-
# @Time  : 2019/12/11 19:52
# @Author : Mr.Lin



'''

可以利用管道将机器学习工作流程中的所有处理步骤封装成一个 scikit-learn 估计
器。这么做的另一个好处在于，现在我们可以使用监督任务（比如回归或分类）的输出来
调节预处理参数。在前几章里，我们在应用岭回归之前使用了 boston 数据集的多项式特
征。下面我们用一个管道来重复这个建模过程。管道包含 3 个步骤：缩放数据、计算多项
式特征与岭回归：
'''

from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target,
random_state=0)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

pipe = make_pipeline(
StandardScaler(),
PolynomialFeatures(),
Ridge())

'''
怎么知道选择几次多项式，或者是否选择多项式或交互项呢？理想情况下，我们希望
根据分类结果来选择 degree 参数。我们可以利用管道搜索 degree 参数以及 Ridge 的 alpha
参数。为了做到这一点，我们要定义一个包含这两个参数的 param_grid ，并用步骤名称作
为前缀：
'''

param_grid = {'polynomialfeatures__degree': [1, 2, 3],
'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}


grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
plt.matshow(grid.cv_results_['mean_test_score'].reshape(3, -1),
vmin=0, cmap="viridis")
plt.xlabel("ridge__alpha")
plt.ylabel("polynomialfeatures__degree")
plt.xticks(range(len(param_grid['ridge__alpha'])), param_grid['ridge__alpha'])
plt.yticks(range(len(param_grid['polynomialfeatures__degree'])),
param_grid['polynomialfeatures__degree'])
plt.colorbar()
# plt.show()


'''
Best parameters: {'polynomialfeatures__degree': 2, 'ridge__alpha': 10}

'''
print("Best parameters: {}".format(grid.best_params_))


































