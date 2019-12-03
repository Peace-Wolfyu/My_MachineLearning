# -*- coding: utf-8 -*-
# @Time  : 2019/12/2 19:12
# @Author : Mr.Lin
from collections import Counter

import numpy as np

class KNN_Model():


    def __init__(self,X_train,y_train,n_neighbors = 3,p = 2):
        """

        :param X_train:  训练集
        :param y_train:  训练集
        :param n_neighbors: 临近点数量  此处默认为3
        :param p:  距离度量 此处使用欧式距离
        """
        self.X_train = X_train

        self.y_train = y_train

        self.neighbors = n_neighbors

        self.p = p



    def prediction(self,X):

        knn_list = []

        # 首先选取前3个点进行计算
        for i in range(self.neighbors):

            # 计算 待分类点与训练集中数据的欧式距离
            distance = np.linalg.norm(X - self.X_train[i],ord=self.p)

            # 将距离 和 对应的类别存在列表中
            knn_list.append((distance,self.y_train[i]))


        # 此时knn_list的数据如下
        # 每个元组里面 前一个数据为距离 后一个数据为对应的类别
        # [(0.8944271909999156, 1.0), (2.209072203437452, 1.0), (0.3999999999999999, 0.0)]

        for i in range(self.neighbors,len(self.X_train)):

            # 首先max函数里面选取出  knn_list 里面最大的距离数据
            # 然后取出它的索引存在 max_index
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))

            # 继续计算待分类点与其他训练集数据的欧式距离
            distance = np.linalg.norm(X - self.X_train[i],ord=self.p)

            # 每次循环迭代knn_list的数据 目标是找到距离小的点
            # 如果新的点比之前列表中的距离要小 那么就添加进去 因为比较的是最大点
            # 要做的就是把距离大的点从列表中剔除出去
            if knn_list[max_index][0] > distance :

                knn_list[max_index] = (distance,self.y_train[i])


        # 解析列表 把对应的类别放入新的列表
        # 此处数据形式
        # [1.0, 1.0, 1.0]
        knn = [k[-1] for k in knn_list]


        #  Counter类的目的是用来跟踪值出现的次数。它是一个无序的容器类型，以字典的键值对形式存储，其中元素作为key
        # 此处就是为了统计每个类别出现的次数，好方便后面最终确定待分类点属于哪一类
        # 对应数据格式
        # Counter({1.0: 3})
        count_pairs = Counter(knn)


        # 最后进行排序 得出出现次数最多的那个类别 作为最终的分类结果
        # 应为 sorted返回的数据格式是列表形式 所以需要进行索引
        # 如  [1.0]
        max_count = sorted(count_pairs, key=lambda x: x)[-1]

        return max_count




    def score(self,X_test,y_test):

        # 分类正确的次数 初始为 0
        right_count = 0

        for X, y in zip(X_test, y_test):
            label = self.prediction(X) # 调用预测模型 得出分类标签

            print("预测类别为：{}".format(label))
            print("实际类别为： {}".format(y))
            print("")

            if label == y: # 如果分类正确
                right_count += 1  #次数 +1


        # 返回最终分类结果表现 正确次数/测试数据集总数
        return right_count / len(X_test)








