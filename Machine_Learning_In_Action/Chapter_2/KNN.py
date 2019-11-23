# -*- coding: utf-8 -*-
# @Time  : 2019/11/23 19:19
# @Author : Mr.Lin


from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array(
        [
            [1.0,1.1],
            [1.0,1.0],
            [0,0],
            [0,0.1]

        ]
    )

    labels = ['A','A','B','B']
    return group,labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]




# 一个分类器
def classify0_details(inX, dataSet, labels, k):

    # 确定数据有多少组
    dataSetSize = dataSet.shape[0]
    print(dataSetSize)
    print("")
    # >>>
    # 4

    # 提供思路 先把每个维度的数据用 待分类数据进行填充
    diffMat_1 = tile(inX, (dataSetSize, 1))
    print(diffMat_1)
    print("")
    # >>>
    # [[0 0]
    #  [0 0]
    #  [0 0]
    #  [0 0]]

    # 根据公式 先把 待分类数据与数据集进行减法操作
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    print(diffMat)
    print("")
    # >>>
    # [[-1. - 1.1]
    #  [-1. - 1.]
    #  [0.   0.]
    # [0. - 0.1]]

    # 对数据进行平方
    sqDiffMat = diffMat**2
    print(sqDiffMat)
    print("")
    # >>>
    # [[1.   1.21]
    #  [1.   1.]
    # [0.
    # 0.]
    # [0.   0.01]]

    # axis=1 将一个矩阵的每一行向量相加
    sqDistances = sqDiffMat.sum(axis=1)
    print(sqDistances)
    print("")
    # >>>
    # [2.21 2.   0.   0.01]

    # 对数据进行开平方操作
    distances = sqDistances**0.5
    print(distances)
    print("")
    # >>>
    # [1.48660687 1.41421356 0.         0.1]

    #argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)
    sortedDistIndicies = distances.argsort()
    print(sortedDistIndicies)
    print("")
    # >>>
    # [2 3 1 0]
    print(distances[2])
    print("")
    # >>>
    # 0.0

    print(distances[0])
    print("")
    # >>>
    # 1.4866068747318506

    # 字典存放结果
    classCount={}


    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        print(voteIlabel)
        print("")
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


group,labels = createDataSet()
print(classify0([0,0],group,labels,3))
# >>>
# B

' 归一化处理 '
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]













