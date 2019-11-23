# -*- coding: utf-8 -*-
# @Time  : 2019/11/23 21:27
# @Author : Mr.Lin


import Machine_Learning_In_Action.Chapter_2.KNN as knn
import Machine_Learning_In_Action.Chapter_2.Action_1 as ac_1


def datingClassTest():
        hoRatio = 0.50  # hold out 10%
        datingDataMat, datingLabels = ac_1.file2matrix('datingTestSet.txt')  # load data setfrom file
        normMat, ranges, minVals = knn.autoNorm(datingDataMat)
        m = normMat.shape[0]
        numTestVecs = int(m * hoRatio)
        errorCount = 0.0
        for i in range(numTestVecs):
            classifierResult = knn.classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
            print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))

            if (classifierResult != datingLabels[i]): errorCount += 1.0
        print(        "the total error rate is: %f" % (errorCount / float(numTestVecs))
)

        print(errorCount)

datingClassTest()









































































