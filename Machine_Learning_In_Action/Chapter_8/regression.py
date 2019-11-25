# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/24 15:40'



from numpy import  *
" 线性回归 "


# 处理数据集
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat


def loadDataSet_detials(fileName):

    numFeat = len(open(fileName).readline().split('\t')) - 1
    print(numFeat)
    print("")
    print("")

    dataMat = []
    labelMat = []


    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            # print(curLine[i])
            lineArr.append(float(curLine[i]))
        # print("=++++++++++++++++++++++")
        # 存放的是 1.0 以及 x 的值
        print(lineArr)
        print("")
        # print("===============================")
        dataMat.append(lineArr)
        print("curline",curLine[-1])
        # curline[-1] 存放  y的值
        labelMat.append(float(curLine[-1]))
    # 最终返回  dataMat：存放 系数和 x的值
    # labelMat： 存放 y的值

    return dataMat,labelMat


def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0 :
        print(" 没有逆矩阵 ")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

def standRegres_details(xArr,yArr):

    xMat = mat(xArr)
    print(xMat)
    print("")
    print("")


    yMat = mat(yArr).T
    print(yMat)
    print("")
    print("")

    xTx = xMat.T * xMat

    # 行列式为 0 的话 就没有可逆矩阵
    if linalg.det(xTx) == 0.0:
        print(" 没有逆矩阵 ")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws



def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print(" 没有逆矩阵 ")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


# 计算误差大小
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

xArr,yArr = loadDataSet('ex0.txt')
# standRegres_details(xArr,yArr)

print(lwlr(xArr[0],xArr,yArr,1.0))
# >>>
# [[3.12204471]]



















































































