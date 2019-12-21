# -*- coding: utf-8 -*-
# @Time  : 2019/12/21 15:31
# @Author : Mr.Lin

'''

加载加州住房价格的数据集

'''

import os
import pandas as pd

def load_housing_data():

    return pd.read_csv("housing.csv")

housing = load_housing_data()


