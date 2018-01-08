import regression
from numpy import *

xArr, yArr = regression.loadDataSet('abalone.txt')
regression.stageWise(xArr, yArr, 0.01, 200)

regression.stageWise(xArr, yArr, 0.001, 5000)

xMat = mat(xArr)
yMat = mat(yArr).T
xMat = regression.regularize(xMat)
yM = mean(yMat, 0)
yMat = yMat - yM
weights = regression.standRegres(xMat, yMat.T)
print(weights.T)
