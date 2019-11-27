# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/27 15:14'



from Machine_Learning_In_Action.Chapter_5 import LogRegres as LogRes

weights = LogRes.gradAscent()
LogRes.plotBestFit(weights=weights.getA())







