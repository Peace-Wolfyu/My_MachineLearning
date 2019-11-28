# -*- coding: utf-8 -*-
# @Time  : 2019/11/28 22:48
# @Author : Mr.Lin


" 决策树 "

from math import log


# 计算给定数据的熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt


# 创建数据
def createDataSet():
    dataSet = [
        [1,1,'yes'],
        [1,0,'yes'],
        [0,1,'no'],
        [0,1,'no']

    ]

    labels = ['no surfacing','flippers']

    return dataSet,labels



# 按照给定特征划分数据集
# 三个参数  待划分的数据集  划分数据集的特征  特征的返回值
def splitDataSet(dataSet,axis,value):

    retDataSet = []

    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet



# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):

    numFeature = len(dataSet[0] - 1)

    baseEntroy = calcShannonEnt(dataSet)

    bestInfoGain = 0.0

    bestFeature = -1

    for i in range(numFeature):

        featList = [example[i] for example in dataSet]

        uniqueVals = set(featList)

        newEntropy = 0.0

        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)

        infoGain = baseEntroy - newEntropy

        if(infoGain > bestInfoGain) :
            bestInfoGain = infoGain

            bestFeature = i

    return bestFeature























