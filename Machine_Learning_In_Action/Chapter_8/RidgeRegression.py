# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/25 15:29'

from numpy import *
import matplotlib.pyplot as plt
from Machine_Learning_In_Action.Chapter_8 import regression as Res

" 岭回归 "


# 计算回归系数
def ridge_regression(xMat,yMat,lam = 0.2):

    xTx = xMat.T * yMat

    denom = xTx + eye(shape(xMat)[1])*lam

    if linalg.det(denom) == 0.0:
        print("无可逆矩阵")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws


# 测试结果
def ridgeTest(xArr,yArr):

    xMat = mat(xArr)

    yMat = mat(yArr).T

    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # to eliminate X0 take mean off of Y
    # regularize X's
    xMeans = mean(xMat, 0)  # calc mean then subtract it off
    xVar = var(xMat, 0)  # calc variance of Xi then divide by it
    xMat = (xMat - xMeans) / xVar

    # 使用30个参数进行测试
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridge_regression(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat




abX,abY = Res.loadDataSet('abalone.txt')

ridgeWeighs = ridgeTest(abX,abY)

fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(ridgeWeighs)

plt.show()








