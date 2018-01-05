from numpy import *
import svmMLiA

dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
b, alphas = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40)

ws = svmMLiA.calcWs(alphas, dataArr, labelArr)
print(ws)

dataMat = mat(dataArr)
print(dataMat[0] * mat(ws) + b)
print(labelArr[0])
print(dataMat[2] * mat(ws) + b)
print(labelArr[2])
print(dataMat[1] * mat(ws) + b)
print(labelArr[1])
