# -*- coding: utf-8 -*-
# @Time  : 2019/11/22 20:39
# @Author : Mr.Lin

from sklearn.datasets import load_breast_cancer
import numpy as np
cancer = load_breast_cancer()

print("cancer.keys(): \n{}".format(cancer.keys()))
# >>>
# cancer.keys():
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
print("")
print("")


' 包含在scikit-learn中的数据集通常保存为Bunch对象' \
'Bunch对象和字典很相似，可以用点操作符号来访问对象的值' \
'如bunch.key 代替  bunch[key] '



print("Shape of cancer data: {}".format(cancer.data.shape))

' 数据集包含569个数据点  每个数据点有30个特征'
# >>>
# Shape of cancer data: (569, 30)
print("")
print("")


print("Sample counts per class : \n{}".format(
    {n:v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}
))
' 569个数据点中 212个被标识为恶性  357个标识为良性'
# >>>
# Sample counts per class :
# {'malignant': 212, 'benign': 357}
print("")
print("")


print("Feature names:\n{}".format(cancer.feature_names))

' 得到每个特征的语义说明'
# >>>
# Feature names:
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']







































