# -*- coding: utf-8 -*-

# @Time  : 2019/12/18 13:34

# @Author : Mr.Lin



'''

LabelEncoder can be used to normalize labels.


'''

from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# le.fit([1, 2, 2, 6])
# LabelEncoder()
# le.classes_
# array([1, 2, 6])
# le.transform([1, 1, 2, 6])
# array([0, 0, 1, 2]...)
# le.inverse_transform([0, 0, 1, 2])
# array([1, 1, 2, 6])


le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])
# LabelEncoder()
list(le.classes_)
# ['amsterdam', 'paris', 'tokyo']
le.transform(["tokyo", "tokyo", "paris"])  # doctest: +ELLIPSIS
# array([2, 2, 1]...)
list(le.inverse_transform([2, 2, 1]))


