# -*- coding: utf-8 -*-
# @Time  : 2019/12/21 16:41
# @Author : Mr.Lin

'''

转换流水线

'''


from sklearn.preprocessing import Imputer, StandardScaler, LabelBinarizer
from Chapter_2.Demo_2.CombinedAttributesAdder import CombinedAttributesAdder
from Chapter_2.Demo_2.Create_Test_Data import housing
from Chapter_2.Demo_2.DataFrameSelector import DataFrameSelector
from sklearn.pipeline import FeatureUnion, Pipeline
from Chapter_2.Demo_2.MyLabelBinarizer import MyLabelBinarizer


housing_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
# print(num_attribs)
cat_attribs = ["ocean_proximity"]


num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])


cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer',MyLabelBinarizer()),
])


full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

# print(housing)
housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared)