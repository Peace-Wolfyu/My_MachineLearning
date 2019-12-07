# -*- coding: utf-8 -*-

# @Time  : 2019/12/7 17:12

# @Author : Mr.Lin





from collections import Counter

import numpy as np
import pandas as pd
import math




good_melon_List = []

bad_melon_List = []

'''
计算均值
'''


def Cal_average(X):
    return sum(X) / float(len(X))


'''
    计算标准差
    '''
def Cal_variance(X):

        avg = Cal_average(X)

        return math.sqrt(sum([pow(x-avg,2) for x in X])) / float(len(X))
# 获取数据集
def create_data():

    # 读取 csv文件
    dataset = pd.read_csv('watermelon_3.csv', delimiter=",")
    del dataset['编号']

    # 获取全部数据
    X = dataset.values[:, :-1]

    # 获取数据的行数和列数
    m, n = np.shape(X)


    # 对数据进行规范，保留三位小数
    for i in range(m):
        X[i, n - 1] = round(X[i, n - 1], 3)
        X[i, n - 2] = round(X[i, n - 2], 3)

    # 获取最后一列数据 即 西瓜所属分类（好瓜 坏瓜）
    y = dataset.values[:, -1]
    for i in range(len(y)):
        if y[i] == '是':
            good_melon_List.append(i)
        else:
            bad_melon_List.append(i)




    return X,y


def Cal_probability(X,attribute,col_id,melon_class,cur_List,the_input_data):

    the_probability = 0
    if col_id >= 6:
        # print("")
        cur_data = X[cur_List,col_id]
        # print("cur_data:\n{}".format(cur_data,melon_class))
        # print("1:   ",Cal_average(cur_data))
        # print("2:   ",Cal_variance(cur_data))
        # print("")
        mean = cur_data.mean()
        # print("mean : \n{}{}{}".format(mean,melon_class,attribute))
        std  = cur_data.std()
        # print("std : \n{}{}{}".format(std,melon_class,attribute))
        # print("")
        # print("the_input_data:\n{}".format(the_input_data))
        # print(type(the_input_data))
        the_input_data = float(the_input_data)
        exponent = math.exp(-(math.pow(the_input_data - mean, 2) / (2 * math.pow(std, 2))))
        the_probability = (1 / (math.sqrt(2 * math.pi) * std)) * exponent
    else:
        for i in cur_List:
            if(X[i,col_id]) == attribute:
                the_probability += 1
        # print("the_probability / len(cur_List)",the_probability,len(cur_List),melon_class)
        the_probability = the_probability / len(cur_List)

    return the_probability

def fit(X_train,y_train,attribute,ColID,X_test):


    # 获取类别列表（用set去除重复类）
    labels = list(set(y_train))
    # print(labels)
    # 构建每个类别对应的数据 用字典的形式
    # 键为类别 值为对应的数据
    # 初始数据格式 {0.0: [], 1.0: []}
    data = {label: [] for label in labels}

    cur_List = []

    for label in labels:
        # print(attribute)
        if label == '是':
            cur_List = good_melon_List
            prob = Cal_probability(X=X_train,attribute=attribute,col_id=ColID,melon_class=label,cur_List=cur_List,the_input_data=X_test[ColID])
            data[label].append(round(prob,3))
        if label == '否':
            cur_List = bad_melon_List
            prob = Cal_probability(X=X_train,attribute=attribute,col_id=ColID,melon_class=label,cur_List=cur_List,the_input_data=X_test[ColID])
            data[label].append(round(prob,3))

    print(*data['是'])











X,y = create_data()
# print(good_melon_List)
# print(bad_melon_List)

# print(X[1,2])
# print(X[:,6])
#
# print("X : \n{}".format(X))
# print("")
# print("")
#
# print("y : \n {}".format(y))



# process_data(X,y)


the_test = np.array(
    ['青绿','蜷缩','浊响','清晰','凹陷','硬滑',0.697,0.460]
)


for i in range(len(the_test)):
    fit(X_train=X,y_train=y,attribute=the_test[i],ColID=i,X_test=the_test)














