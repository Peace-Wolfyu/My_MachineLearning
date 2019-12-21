# -*- coding: utf-8 -*-
# @Time  : 2019/12/21 16:00
# @Author : Mr.Lin

'''

尽管sklearn提供了强大的数据处理功能，有些时候我们需要根据自己的需求自定义一些数据预处理方法，并且让我们这些操作有着sklearnAPI相似的用法，
我们所需要做的就是继承BaseEstimator类，并覆写三个方法fit，transform和fit_transform，第三个方法是前两个的整合，
如果不想覆写fit_transform,可以继承TransformerMixin(从类名称就可以看出其作用)这个类

转换器有一个超参数add_bedrooms_per_room默认设置为True（提供合理的默认值通常是很有帮助的）。这个超参数可以让你轻松知晓添加这个属性是否有助于机器学习的算法。更广泛地
说，如果你对数据准备的步骤没有充分的信心，就可以添加这个超参数来进行把关。这些数据准备步骤的执行越自动化，你自动尝试的组合也就越多，
从而有更大可能从中找到一个重要的组合（还节省了大量时间）。

'''


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


# 类的功能是为原数据集添加新的特征，X[:,3]表示的是第4列所有数据，np.c_表示的是拼接数组。
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# 使用方法
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

# housing_extra_attribs = attr_adder.transform(housing.values)



