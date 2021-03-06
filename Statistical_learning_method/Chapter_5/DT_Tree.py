# -*- coding: utf-8 -*-

# @Time  : 2019/12/6 16:26

# @Author : Mr.Lin

"""
决策树
•ID3（基于信息增益）
•C4.5（基于信息增益比）
•CART（gini指数）
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math
from math import log

import pprint


# 书上题目5.1
def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels




# print(datasets)
# print("")
# print(labels)
#
# train_data = pd.DataFrame(datasets, columns=labels)

# print(train_data)

# 熵
def calc_ent(datasets):
    data_length = len(datasets)
    label_count = {}
    for i in range(data_length):
        label = datasets[i][-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    ent = -sum([(p/data_length)*log(p/data_length, 2) for p in label_count.values()])
    return ent

# 经验条件熵
def cond_ent(datasets, axis=0):
    data_length = len(datasets)
    feature_sets = {}
    for i in range(data_length):
        feature = datasets[i][axis]
        if feature not in feature_sets:
            feature_sets[feature] = []
        feature_sets[feature].append(datasets[i])
    cond_ent = sum([(len(p)/data_length)*calc_ent(p) for p in feature_sets.values()])
    return cond_ent

# 信息增益
def info_gain(ent, cond_ent):
    return ent - cond_ent

def info_gain_train(datasets):
    count = len(datasets[0]) - 1
    ent = calc_ent(datasets)
    best_feature = []
    for c in range(count):
        c_info_gain = info_gain(ent, cond_ent(datasets, axis=c))
        best_feature.append((c, c_info_gain))
        print('特征({}) - info_gain - {:.3f}'.format(labels[c], c_info_gain))
    # 比较大小
    best_ = max(best_feature, key=lambda x: x[-1])
    return '特征({})的信息增益最大，选择为根节点特征'.format(labels[best_[0]])

# print(info_gain_train(np.array(datasets)))


# 定义节点类 二叉树
class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {'label:': self.label, 'feature': self.feature, 'tree': self.tree,'feature_name':self.feature_name}

    def __repr__(self):
        return '++{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)


class DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    # 熵
    @staticmethod
    def calc_ent(datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(p / data_length) * log(p / data_length, 2) for p in label_count.values()])
        return ent

    # 经验条件熵
    def cond_ent(self, datasets, axis=0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum([(len(p) / data_length) * self.calc_ent(p) for p in feature_sets.values()])
        return cond_ent

    # 信息增益
    @staticmethod
    def info_gain(ent, cond_ent):
        return ent - cond_ent

    def info_gain_train(self, datasets):
        count = len(datasets[0]) - 1
        ent = self.calc_ent(datasets)
        best_feature = []
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(datasets, axis=c))
            best_feature.append((c, c_info_gain))
        # 比较大小
        best_ = max(best_feature, key=lambda x: x[-1])
        return best_

    def train(self, train_data):
        """
        input:数据集D(DataFrame格式)，特征集A，阈值eta
        output:决策树T
        """
        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[:-1]


        print("_                    ",_)
        print("")
        print("(y_train          ",y_train)

        print("")

        print("features,",features)
        print("+++++")
        # 1,若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        if len(y_train.value_counts()) == 1:
            print("label=y_train.iloc[0]\n{}".format(y_train.iloc[0]))
            return Node(root=True,
                        label=y_train.iloc[0])



        # 2, 若A为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
        if len(features) == 0:

            print("len(features) == 0",features)
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 3,计算最大信息增益 同5.1,Ag为信息增益最大的特征
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]
        print("max_name    max_info_gain  {}{}".format(features[max_feature],max_info_gain))

        # 4,Ag的信息增益小于阈值eta,则置T为单节点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
        if max_info_gain < self.epsilon:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0],feature_name=max_feature_name)

        # 5,构建Ag子集
        node_tree = Node(root=False, feature_name=max_feature_name, feature=max_feature)

        feature_list = train_data[max_feature_name].value_counts().index
        print("feature_list  {}".format(feature_list))
        print("")

        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)


            # 6, 递归生成树
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)

        # pprint.pprint(node_tree.tree)
        return node_tree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, X_test):
        return self._tree.predict(X_test)

def create__data__melon():

    # 读取 csv文件
    dataset = pd.read_csv('watermelon_3.csv', delimiter=",")
    del dataset['编号']
    # print(*dataset.columns)
    y = dataset.columns.tolist()

    # 获取全部数据
    X = dataset.values[:, :]

    print("X：\n{}".format(X))


    return X.tolist(),y

# datasets, labels = create_data()



datasets, labels = create__data__melon()
print(type(datasets))
print(datasets)

print(labels)

data_df = pd.DataFrame(datasets, columns=labels)
dt = DTree()
tree = dt.fit(data_df)

print(tree)

# {'label:':
# None,
# 'feature':
# 2,
# 'tree':
# {'否':
# {'label:': None, 'feature': 1,
# 'tree': {'否': {'label:': '否', 'feature': None,
# 'tree': {}}, '是': {'label:': '是', 'feature': None,
# 'tree': {}}}}, '是': {'label:': '是', 'feature': None, 'tree': {}}}}


# print(dt.predict(['老年', '否', '否', '一般']))


# ++{'label:': None, 'feature': 3,
# 'tree': {'清晰': ++{'label:': None, 'feature': 1,
# 'tree': {'蜷缩': ++{'label:': '是', 'feature': None,
# 'tree': {}, 'feature_name': None}, '稍蜷': ++{'label:': None, 'feature': 0,
# 'tree': {'乌黑': ++{'label:': None, 'feature': 2,
# 'tree': {'软粘': ++{'label:': '否', 'feature': None,
# 'tree': {}, 'feature_name': None}, '硬滑': ++{'label:': '是', 'feature': None,
# 'tree': {}, 'feature_name': None}}, 'feature_name': '触感'}, '青绿': ++{'label:': '是', 'feature': None,
# 'tree': {}, 'feature_name': None}}, 'feature_name': '色泽'}, '硬挺': ++{'label:': '否', 'feature': None,
# 'tree': {}, 'feature_name': None}}, 'feature_name': '根蒂'}, '稍糊': ++{'label:': None, 'feature': 4,
# 'tree': {'硬滑': ++{'label:': '否', 'feature': None,
# 'tree': {}, 'feature_name': None}, '软粘': ++{'label:': '是', 'feature': None,
# 'tree': {}, 'feature_name': None}}, 'feature_name': '触感'}, '模糊': ++{'label:': '否', 'feature': None,
# 'tree': {}, 'feature_name': None}}, 'feature_name': '纹理'}