# -*- coding: utf-8 -*-
# @Time  : 2019/12/5 20:11
# @Author : Mr.Lin

"""
prior probability


"""
from collections import Counter
import numpy as np
import math
class NaiveBayes():

    def __init__(self):

        # 先验概率
        self.prior_probability = None

        # 最终的数据，用以计算最后的概率
        self.model_data = None



    '''
    计算先验概率
    '''
    def Cal_prior_probability(self,y):

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

        self.prior_probability = label



    '''
    计算均值
    '''
    def Cal_average(self,X):
        return sum(X)/float(len(X))


    '''
    计算标准差
    '''
    def Cal_variance(self,X):

        avg = self.Cal_average(X)

        return math.sqrt(sum([pow(x-avg,2) for x in X])) / float(len(X))


    '''
    概率密度函数x  假设服从高斯分布
    '''
    def Cal_gaussian_probability(self,x,mean,variance):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(variance, 2))))
        return (1 / (math.sqrt(2 * math.pi) * variance)) * exponent


    '''
    处理训练数据
    计算训练数据中每一列特征的均值和标准差
    '''
    def Process_train_data(self,train_data):

        # 分别计算数据集中每列数据的均值和标准差
        # 数据形式 [(5.9378378378378383, 0.08280609606147994), (2.7783783783783784, 0.050292807376915244), (4.2918918918918916, 0.0773349349284356), (1.3405405405405406, 0.030687153337797462)]
        the_data = [(self.Cal_average(i),self.Cal_variance(i)) for i in zip(*train_data)]

        return the_data


    def fit(self,X,y):

        # 获取类别列表（用set去除重复类）
        labels = list(set(y))

        # 构建每个类别对应的数据 用字典的形式
        # 键为类别 值为对应的数据
        # 初始数据格式 {0.0: [], 1.0: []}
        data = {label:[] for label in labels}

        # 在相应类别添加数据
        # 处理后格式 {0.0: [array([ 5. ,  3.2,  1.2,  0.2]), array([ 5.2,  4.1,  1.5,  0.1]), array([ 5.1,  3.8,  1.5,  0.3]), array([ 5.4,  3.4,  1.7,  0.2]), array([ 4.8,  3.4,  1.6,  0.2]), array([ 4.8,  3.1,  1.6,  0.2]), array([ 4.9,  3.1,  1.5,  0.1]), array([ 5.4,  3.9,  1.3,  0.4]), array([ 5.1,  3.5,  1.4,  0.3]), array([ 4.6,  3.6,  1. ,  0.2]), array([ 5.4,  3.9,  1.7,  0.4]), array([ 5.1,  3.8,  1.9,  0.4]), array([ 5.3,  3.7,  1.5,  0.2]), array([ 4.7,  3.2,  1.3,  0.2]), array([ 5.1,  3.3,  1.7,  0.5]), array([ 4.4,  3.2,  1.3,  0.2]), array([ 5.5,  4.2,  1.4,  0.2]), array([ 5. ,  3. ,  1.6,  0.2]), array([ 5.2,  3.4,  1.4,  0.2]), array([ 5.1,  3.4,  1.5,  0.2]), array([ 5.7,  4.4,  1.5,  0.4]), array([ 4.8,  3. ,  1.4,  0.3]), array([ 5.1,  3.8,  1.6,  0.2]), array([ 4.6,  3.2,  1.4,  0.2]), array([ 4.6,  3.1,  1.5,  0.2]), array([ 4.6,  3.4,  1.4,  0.3]), array([ 4.9,  3.1,  1.5,  0.1]), array([ 5. ,  3.5,  1.6,  0.6]), array([ 5.7,  3.8,  1.7,  0.3]), array([ 5. ,  3.5,  1.3,  0.3]), array([ 5.5,  3.5,  1.3,  0.2]), array([ 4.9,  3. ,  1.4,  0.2]), array([ 4.4,  2.9,  1.4,  0.2]), array([ 5. ,  3.6,  1.4,  0.2]), array([ 5. ,  3.4,  1.5,  0.2]), array([ 4.8,  3. ,  1.4,  0.1])], 1.0: [array([ 6. ,  2.9,  4.5,  1.5]), array([ 5.9,  3. ,  4.2,  1.5]), array([ 5.5,  2.6,  4.4,  1.2]), array([ 5.9,  3.2,  4.8,  1.8]), array([ 5.6,  2.9,  3.6,  1.3]), array([ 5.7,  2.9,  4.2,  1.3]), array([ 6.4,  2.9,  4.3,  1.3]), array([ 5.7,  2.6,  3.5,  1. ]), array([ 5.8,  2.7,  3.9,  1.2]), array([ 6.8,  2.8,  4.8,  1.4]), array([ 6.6,  2.9,  4.6,  1.3]), array([ 6.3,  3.3,  4.7,  1.6]), array([ 6. ,  3.4,  4.5,  1.6]), array([ 5. ,  2.3,  3.3,  1. ]), array([ 6.6,  3. ,  4.4,  1.4]), array([ 6.3,  2.3,  4.4,  1.3]), array([ 6.7,  3.1,  4.4,  1.4]), array([ 6.2,  2.2,  4.5,  1.5]), array([ 5.6,  2.7,  4.2,  1.3]), array([ 6.3,  2.5,  4.9,  1.5]), array([ 6.9,  3.1,  4.9,  1.5]), array([ 6.1,  3. ,  4.6,  1.4]), array([ 6.7,  3. ,  5. ,  1.7]), array([ 6.7,  3.1,  4.7,  1.5]), array([ 5.5,  2.5,  4. ,  1.3]), array([ 5. ,  2. ,  3.5,  1. ]), array([ 5.4,  3. ,  4.5,  1.5]), array([ 5.6,  3. ,  4.1,  1.3]), array([ 5.5,  2.4,  3.7,  1. ]), array([ 5.2,  2.7,  3.9,  1.4]), array([ 6.1,  2.9,  4.7,  1.4]), array([ 5.8,  2.7,  4.1,  1. ]), array([ 5.7,  3. ,  4.2,  1.2]), array([ 5.5,  2.3,  4. ,  1.3])]}
        for f,label in zip(X,y):
            data[label].append(f)

        # 计算先验概率
        self.Cal_prior_probability(y)

        # 计算训练数据的均值以及标准差 并且用字典的格式与对应的类别对应起来
        # 最终数据格式  {0.0: [(5.0277777777777786, 0.05978499428651085), (3.4249999999999998, 0.06530732191949305), (1.4500000000000002, 0.03130396574884623), (0.24999999999999994, 0.01643355495305449)],
        # 1.0: [(5.9176470588235288, 0.08539249785075936), (2.7794117647058827, 0.0484677704357821), (4.2882352941176478, 0.07856462152744104), (1.3411764705882352, 0.03225054573150113)]}
        self.model_data = {label:self.Process_train_data(value) for label,value in data.items()}

        return "Model Data trained Done"


    # 计算概率
    def Cal_probabilities(self,the_test_data):

        # 存放最后的概率 用字典的格式 键 类别  值  最终的概率
        probabilities = {}

        for label,value in self.model_data.items():

            # 取出对应类别的先验概率
            probabilities[label] = self.prior_probability[label]

            for i in range(len(value)):

                # 取出对应的均值以及标准差 用于后面概率的计算
                average,variance = value[i]

                # 根据公式 计算最后的贝叶斯概率
                probabilities[label] *= self.Cal_gaussian_probability(the_test_data[i],average,variance)

        # 最终返回的数据格式  {0.0: 2.9634275875804e-31, 1.0: 0.0}
        print(" probabilities\n{}".format(probabilities))
        print("")
        return probabilities


    def prediction(self,X_test):

        # 排序之后的数据格式如下：
        # [(1.0, 0.0), (0.0, 7.78343549675974e-36)] 元组数据里面前者为类别 后者为概率
        # 因为最终的数据是按照概率从小到大的顺序，所以进行索引选择最后面那个的类别作为最终的预测类别

        label = sorted(self.Cal_probabilities(X_test).items(), key=lambda x: x[-1])[-1][0]
        return label










































































































































