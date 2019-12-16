# -*- coding: utf-8 -*-

# @Time  : 2019/12/16 12:52

# @Author : Mr.Lin


'''

用于回归分析的波士顿房价数据集（Boston）

'''


from sklearn import datasets
from numpy import shape
loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

# (506, 13)

# (506,)

# [[6.3200e-03 1.8000e+01 2.3100e+00 0.0000e+00 5.3800e-01 6.5750e+00
#   6.5200e+01 4.0900e+00 1.0000e+00 2.9600e+02 1.5300e+01 3.9690e+02
#   4.9800e+00]
#  [2.7310e-02 0.0000e+00 7.0700e+00 0.0000e+00 4.6900e-01 6.4210e+00
#   7.8900e+01 4.9671e+00 2.0000e+00 2.4200e+02 1.7800e+01 3.9690e+02
#   9.1400e+00]]


# [24.  21.6]
# print(shape(data_X))
# print(shape(data_y))
# print(data_X[:2, :])
# print(data_y[:2])


'''
Keys of iris_dataset : 
 dict_keys(['data', 'target', 'feature_names', 'DESCR'])
'''
print(" Keys of loaded_data : \n {}".format(loaded_data.keys()))

print("")
print("")


'''
 feature_names of loaded_data : 
 ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 'B' 'LSTAT']
'''
print(" feature_names of loaded_data : \n {}".format(loaded_data['feature_names']))

print("")
print("")

# print(" target of loaded_data : \n {}".format(loaded_data['target']))


'''
 DESCR of loaded_data : 
 Boston House Prices dataset
===========================

Notes
------
Data Set Characteristics:  

    :Number of Instances: 506 

    :Number of Attributes: 13 numeric/categorical predictive
    
    :Median Value (attribute 14) is usually the target

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

    :Missing Attribute Values: None

    :Creator: Harrison, D. and Rubinfeld, D.L.

This is a copy of UCI ML housing dataset.
http://archive.ics.uci.edu/ml/datasets/Housing


This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
prices and the demand for clean air', J. Environ. Economics & Management,
vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
...', Wiley, 1980.   N.B. Various transformations are used in the table on
pages 244-261 of the latter.

The Boston house-price data has been used in many machine learning papers that address regression
problems.   
     
**References**

   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
   - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)

'''
# print(" DESCR of loaded_data : \n {}".format(loaded_data['DESCR']))


'''
data of loaded_data : 
 (506, 13)
'''
print(" data of loaded_data : \n {}".format(loaded_data['data'].shape))

