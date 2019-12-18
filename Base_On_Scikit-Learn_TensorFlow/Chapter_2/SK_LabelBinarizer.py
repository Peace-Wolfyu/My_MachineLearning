# -*- coding: utf-8 -*-

# @Time  : 2019/12/18 13:59

# @Author : Mr.Lin
from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()
trans = lb.fit_transform(['yes', 'no', 'no', 'yes'])
print(type(trans))


'''
<class 'numpy.ndarray'>

'''
'''
[[1]
 [0]
 [0]
 [1]]
'''









