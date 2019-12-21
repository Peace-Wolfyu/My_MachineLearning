# -*- coding: utf-8 -*-
# @Time  : 2019/12/21 15:54
# @Author : Mr.Lin

'''

创建测试集

'''

from sklearn.model_selection import train_test_split
from Chapter_2.Demo_2.DataSource import housing
import numpy as np

# Scikit-Learn提供了一些函数，可以通过多种方式将数据集分成多
# 个子集。最简单的函数是train_test_split，它与前面定义的函数
# split_train_test几乎相同，除了几个额外特征。首先，它也有
# random_state参数，让你可以像之前提到过的那样设置随机生成器种
# 子；其次，你可以把行数相同的多个数据集一次性发送给它，它会根
# 据相同的索引将其拆分

# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Divide by 1.5 to limit the number of income categories
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# 根据收入类别进行分层抽样了。使用Scikit-Learn的
# Stratified-Shuffle Split类

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set


# print(housing)
housing_labels = strat_train_set["median_house_value"].copy()
# print("")
# print(housing_labels)




