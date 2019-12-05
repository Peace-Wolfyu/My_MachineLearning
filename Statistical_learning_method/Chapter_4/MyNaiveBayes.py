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
        cnt = Counter(y)
        label = {i: {} for i in cnt.keys()}
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
    概率密度函数x
    '''
    def Cal_gaussian_probability(self,x,mean,variance):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(variance, 2))))
        return (1 / (math.sqrt(2 * math.pi) * variance)) * exponent




    '''
    处理训练数据
    计算训练数据中每一列特征的均值和标准差
    '''
    def Process_train_data(self,train_data):

        the_data = [(self.Cal_average(i),self.Cal_variance(i)) for i in zip(*train_data)]

        return the_data




    def fit(self,X,y):

        # 获取类别列表（用set去除重复类）
        labels = list(set(y))

        # 构建每个类别对应的数据 用字典的形式
        # 键为类别 值为对应的数据
        data = {label:[] for label in labels}

        # 在相应类别添加数据
        for f,label in zip(X,y):
            data[label].append(f)


        self.Cal_prior_probability(y)
        self.model_data = {label:self.Process_train_data(value) for label,value in data.items()}

        return "Model Data trained Done"


    # 计算概率
    def Cal_probabilities(self,the_test_data):

        probabilities = {}

        for label,value in self.model_data.items():

            probabilities[label] = self.prior_probability[label]

            for i in range(len(value)):

                average,variance = value[i]

                probabilities[label] *= self.Cal_gaussian_probability(the_test_data[i],average,variance)



        return probabilities


    def prediction(self,X_test):
        # self.Cal_probabilities(X_test).items()
        label = sorted(self.Cal_probabilities(X_test).items(), key=lambda x: x[-1])[-1][0]
        return label










































































































































