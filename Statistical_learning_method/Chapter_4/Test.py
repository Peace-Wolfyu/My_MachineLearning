# -*- coding: utf-8 -*-
# @Time  : 2019/12/5 20:47
# @Author : Mr.Lin

import numpy as np


data = [
    np.array([1,2,3]),
    np.array([4,5,6]),
    np.array([7,8,9]),

]

for i in zip(data):
    print(" i : \n{}".format(i))


#  i :
# (array([1, 2, 3]),)
#  i :
# (array([4, 5, 6]),)
#  i :
# (array([7, 8, 9]),)



for i in zip(*data):
    print(" i : \n{}".format(i))

#  i :
# (1, 4, 7)
#  i :
# (2, 5, 8)
#  i :
# (3, 6, 9)


def fun(x,y):
    return x+y
print(fun(1))
