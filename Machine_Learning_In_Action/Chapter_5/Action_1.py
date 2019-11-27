# -*- coding: utf-8 -*-
# @Time  : 2019/11/27 20:35
# @Author : Mr.Lin



from numpy import *



" 预测病马的死亡率 "

# sigmoid函数
def sigmoid(intX):
    return 1.0/(1+exp(-intX))



def classifyVector(intX,weights):
    prob = sigmoid(sum(intX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        # dataIndex = range(m)
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def colicTest():
        frTrain = open('horseColicTraining.txt')
        frTest = open('horseColicTest.txt')
        trainingSet = []
        trainingLabels = []
        for line in frTrain.readlines():
            currLine = line.strip().split('\t')
            lineArr = []
            for i in range(21):
                lineArr.append(float(currLine[i]))
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[21]))
        trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
        errorCount = 0;
        numTestVec = 0.0
        for line in frTest.readlines():
            numTestVec += 1.0
            currLine = line.strip().split('\t')
            lineArr = []
            for i in range(21):
                lineArr.append(float(currLine[i]))
            if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
                errorCount += 1
        errorRate = (float(errorCount) / numTestVec)
        print("the error rate of this test is: %f" % errorRate)
        return errorRate




def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))


multiTest()































