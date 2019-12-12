# -*- coding: utf-8 -*-

# @Time  : 2019/12/7 17:12

# @Author : Mr.Lin

"""
根据西瓜书中的例子进行测试（朴素贝叶斯）
"""

from collections import Counter
import numpy as np
import pandas as pd
import math

# 全局变量 存放 好瓜的索引位置
good_melon_List = []

# 全局变量 存放 坏瓜的索引位置
bad_melon_List = []

# 全局变量 存放 基于好瓜的后验概率
the_good_melon_probability = []

# 全局变量 存放 基于坏瓜的后验概率
the_bad_melon_probability = []


# 获取数据集以及处理
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

    # 获取好瓜 坏瓜的位置
    for i in range(len(y)):
        if y[i] == '是':
            good_melon_List.append(i)
        else:
            bad_melon_List.append(i)
    return X,y


# 计算概率
def Cal_probability(X,attribute,col_id,melon_class,cur_List,the_input_data):
    """

    :param X: 训练数据
    :param attribute: 待分类中的特征属性
    :param col_id: 列id
    :param melon_class: 瓜的类别（好 坏）
    :param cur_List:（当前处理的数据列表（里面全是好瓜或者全是坏瓜））
    :param the_input_data:（输入待预测的数据）
    :return:
    """

    # 初始化概率
    the_probability = 0

    # 根据西瓜数据特点  列id在6之后的数据为连续型数据 所以需要概率密度函数来处理
    if col_id >= 6:

        # 当前处理的列数据（需要区分好瓜还是坏瓜）
        cur_data = X[cur_List,col_id]

        # 求均值
        mean = cur_data.mean()

        # 求标准差
        std  = cur_data.std()

        # 把待分类数据转换为浮点数 用以后续的计算
        the_input_data = float(the_input_data)

        # 计算概率密度函数
        exponent = math.exp(-(math.pow(the_input_data - mean, 2) / (2 * math.pow(std, 2))))

        the_probability = (1 / (math.sqrt(2 * math.pi) * std)) * exponent
    # 针对离散数据的处理
    else:

        # 遍历当前处理的所有好瓜或者坏瓜数据
        for i in cur_List:

            # 出现相同属性则对应概率+1
            if(X[i,col_id]) == attribute:
                the_probability += 1

        the_probability = the_probability / len(cur_List)

    return the_probability

# 训练数据
def fit(X_train,y_train,attribute,ColID,X_test):

    """

    :param X_train:  训练数据
    :param y_train:  训练数据
    :param attribute: 待分类属性
    :param ColID:  列id
    :param X_test: 测试数据
    :return:
    """

    # 获取类别列表（用set去除重复类）
    labels = list(set(y_train))

    # 构建每个类别对应的数据 用字典的形式
    # 键为类别 值为对应的数据
    # 初始数据格式 {0.0: [], 1.0: []}
    data = {label: [] for label in labels}

    cur_List = []

    for label in labels:

        # 计算好瓜的各个特征概率
        if label == '是':
            cur_List = good_melon_List
            prob = Cal_probability(X=X_train,attribute=attribute,col_id=ColID,melon_class=label,cur_List=cur_List,the_input_data=X_test[ColID])
            data[label].append(round(prob,3))

        # 计算坏瓜的各个特征概率
        if label == '否':
            cur_List = bad_melon_List
            prob = Cal_probability(X=X_train,attribute=attribute,col_id=ColID,melon_class=label,cur_List=cur_List,the_input_data=X_test[ColID])
            data[label].append(round(prob,3))

    # 分别把概率添加到全局变量中
    the_good_melon_probability.append(*data['是'])
    the_bad_melon_probability.append(*data['否'])



# 最终的预测
def prediction():

    good = 0
    bad = 0
    for i in the_good_melon_probability:
        good = math.log(i) + good

    for j in the_bad_melon_probability:
        bad = math.log(j) + bad

    print(good)
    print(bad)
    if good > bad:
        return '好'
    else:
        return '否'

X,y = create_data()

the_test = np.array(
    ['青绿','蜷缩','浊响','清晰','凹陷','硬滑',0.697,0.460]
)

for i in range(len(the_test)):
    fit(X_train=X,y_train=y,attribute=the_test[i],ColID=i,X_test=the_test)

print(prediction())












