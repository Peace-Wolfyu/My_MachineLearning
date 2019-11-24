# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/24 15:49'



import Machine_Learning_In_Action.Chapter_8.regression as regression
import matplotlib.pyplot as plt
from numpy import *

xArr,yArr = regression.loadDataSet('ex0.txt')

print(xArr[0:2])
# >>>
# [[1.0, 0.067732], [1.0, 0.42781]]
print("")
print("")

ws = regression.standRegres(xArr,yArr)
print(ws)
# >>>
# [[3.00774324]
#  [1.69532264]]


xMat = mat(xArr)
yMat = mat(yArr)
yHat = xMat * ws

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:,1],yHat)
plt.show()


















