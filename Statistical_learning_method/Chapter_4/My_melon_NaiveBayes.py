# -*- coding: utf-8 -*-

# @Time  : 2019/12/7 14:15

# @Author : Mr.Lin
from collections import Counter

import numpy as np
import pandas as pd
import math



good_melon_List = []

bad_melon_List = []

probability_Map = {}

kindsOfAttribute={}

store_Mean_And_Variance = {}


def prepare_Data(X_train,y_train):

    for i in range(len(y_train)):
        if y_train[i] == '是':
            good_melon_List.append(i)
        else:
            bad_melon_List.append(i)





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

'''
    概率密度函数x  假设服从高斯分布
'''
def Cal_gaussian_probability(x,mean,variance):

        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(variance, 2))))
        return (1 / (math.sqrt(2 * math.pi) * variance)) * exponent


'''
计算先验概率
'''
def Cal_prior_probability(y):

        # 计算每个类别所出现的次数
        # 数据形式 Counter({0.0: 36, 1.0: 34})
        cnt = Counter(y)

        # 用字典的形式处理数据  键为 类别  值为 对应的概率 此处只是初始化 所以概率为空
        # 数据形式 {0.0: {}, 1.0: {}}
        label = {i: {} for i in cnt.keys()}

        # 计算每个类别的先验概率
        # 最终的数据形式 ： {1.0: 0.4857142857142857, 0.0: 0.5142857142857142}
        for key in cnt.keys():
            label[key] = cnt[key] / len(y)

        return label
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

    for i in range(n):
        kindsOfAttribute[i] = len(set(X[:, i]))


    return X,y


def process_data(X_train,y_train):
    # 获取类别列表（用set去除重复类）
    labels = list(set(y_train))
    # print(labels)
    # 构建每个类别对应的数据 用字典的形式
    # 键为类别 值为对应的数据
    # 初始数据格式 {0.0: [], 1.0: []}
    data = {label: [] for label in labels}

    # 在相应类别添加数据
    for f, label in zip(X_train, y_train):
        data[label].append(f)

    # print(data)

    for label,value in data.items():
        print(label)
        the_data = {label: []}
        # print("")
        for i in zip(*value):
            cnt = Counter(i)
            for key in cnt.keys():
                # print("len",len(i))
                the_data[label].append(cnt[key] / len(i))
                # print(cnt[key] / len(i))
        print(the_data)
    # print(the_data)


def Cal_probability(ColmnID,attribute,melon_Class,X_train):


    current_List = []

    the_probability = 0

    if melon_Class == '是':
        current_List = good_melon_List
    else:
        current_List = bad_melon_List


    if ColmnID >= 6 :
        # print("6")

        mean = 1
        Variance = 1

        if (ColmnID,melon_Class) in store_Mean_And_Variance:

            current_Data = store_Mean_And_Variance[(ColmnID,melon_Class)]

            mean = current_Data[0]

            Variance = current_Data[1]


        else:

            current_Data = X_train[current_List,ColmnID]
            print("current_Data:\n{}".format(current_Data))
            # mean = Cal_average(current_Data)
            mean = current_Data.mean()
            # Variance = Cal_variance(current_Data)
            Variance = current_Data.std()
            store_Mean_And_Variance[(ColmnID,melon_Class)] = (mean,Variance)

        the_probability =         ans=1/(math.sqrt(math.pi*2)*Variance)*math.exp((-(attribute-mean)**2)/(2*Variance*Variance))# 高斯分布

        print(the_probability)

    else:
        for i in current_List:
            if X_train[i,ColmnID] == attribute:
                the_probability += 1
        print(attribute)
        the_probability = (the_probability+1) / (len(current_List) + kindsOfAttribute[ColmnID])
        print(the_probability)

    probability_Map[ColmnID,attribute,melon_Class] = the_probability


    return the_probability



def prediction(X_train,X_test,y_train):


    the_good_melon_probability = math.log((len(good_melon_List)+1) / (len(y_train)+2))


    the_bad_melon_probability =  math.log((len(bad_melon_List)+1) / (len(y_train)+2))

    # print("ppp")
    print(len(X_test))
    for i in range(len(X_test)):
        # print(i)
        the_good_melon_probability += math.log(Cal_probability(i, X_test[i], '是',X_train))
        print("*******************")
        the_bad_melon_probability += math.log(Cal_probability(i, X_test[i], '否',X_train))

    if the_good_melon_probability > the_bad_melon_probability :
            return "good"
    else:
            return "bad"

X_train,y_train = create_data()

# print(Cal_prior_probability(y_train))
# process_data(X_train,y_train)


prepare_Data(X_train,y_train)
the_test = np.array(
    ['青绿','蜷缩','浊响','清晰','凹陷','硬滑',0.697,0.460]
)


print(prediction(X_train,the_test,y_train))









































