# -*- coding: utf-8 -*-
# @Time  : 2019/11/12 20:37
# @Author : Mr.Lin


"""

DataFrame的使用

"""

import pandas as pd

data = {
    'country':['china','russian','janpan','usa','uk'],
    'year':[2001,2002,2003,2004,2005],
    'alpha':['a','b','c','d','e']

}

frame = pd.DataFrame(data)

print(frame)

print("==============================")

"""
指定列序列
"""

frame_1 = pd.DataFrame(data,columns=['year','alpha','country'])

print(frame_1)
print("==============================")


"""
传入的列找不到就会产生缺失值
"""

frame_2 = pd.DataFrame(data,
                       columns=['year','country','alpha','aaaa'])

print(frame_2)
print("==============================")



"""
通过类似字典标记以及属性方式，可以将DataFrame列 获取成 一个Series
"""

print(frame_1['year'])
print("==============================")


"""
嵌套字典

外层字典的键作为列

内层键作为行索引
"""

pop = {

    'num':{
        20001:100,
        200022:102929,
        2388383:11
    },
    'aossp':{
        2000:789384,
        38848548:895823458
    }
}

frame_3 = pd.DataFrame(pop)

print(frame_3)