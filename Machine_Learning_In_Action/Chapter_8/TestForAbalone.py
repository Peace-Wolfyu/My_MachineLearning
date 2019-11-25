# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/25 14:56'

" 预测鲍鱼年龄 "

from Machine_Learning_In_Action.Chapter_8 import regression as regression


abX,abY = regression.loadDataSet('abalone.txt')

yHat_01 = regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)

yHat_02 = regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)

yHat_03 = regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)

print(regression.rssError(abY[0:99],yHat_01.T))
# >>>
# 56.83512842240429

print("")
print("")

print(regression.rssError(abY[0:99],yHat_02.T))
# >>>
# 429.8905618701634

print("")
print("")

print(regression.rssError(abY[0:99],yHat_03.T))
# >>>
# 549.1181708823923

' 新数据的表现 '

yHat_001 = regression.lwlrTest(abY[100:199],abX[0:99],abY[0:99],0.1)

print(regression.rssError(abY[100:199],yHat_001.T))



