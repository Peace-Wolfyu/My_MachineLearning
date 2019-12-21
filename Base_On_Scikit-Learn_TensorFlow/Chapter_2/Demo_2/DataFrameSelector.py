# -*- coding: utf-8 -*-
# @Time  : 2019/12/21 15:56
# @Author : Mr.Lin


'''

另外sklearn是不能直接处理DataFrames的，那么我们需要自定义一个处理的方法将之转化为numpy类型

'''

from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

