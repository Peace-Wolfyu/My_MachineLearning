# -*- coding: utf-8 -*-
# @Time  : 2019/12/21 17:06
# @Author : Mr.Lin

'''

针对错误 TypeError: fit_transform() takes 2 positional arguments but 3 were given的解决方案

'''


from sklearn.base import TransformerMixin #gives fit_transform method for free
from sklearn.preprocessing import LabelBinarizer


class MyLabelBinarizer(TransformerMixin):

    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x)



