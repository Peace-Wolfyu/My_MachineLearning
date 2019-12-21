# -*- coding: utf-8 -*-
# @Time  : 2019/12/21 17:19
# @Author : Mr.Lin

'''

开始训练模型

'''


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from Chapter_2.Demo_2.Create_Test_Data import housing_labels, housing
from Chapter_2.Demo_2.Transform_Pipeline import housing_prepared, full_pipeline

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
# print(some_data)
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
# 简单进行预测
# print("Predictions:", lin_reg.predict(some_data_prepared))
# print("")
# print("Labels:", list(some_labels))

# 用Scikit-Learn的mean_squared_error函数来
# 测量整个训练集上回归模型的RMSE：

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)


















