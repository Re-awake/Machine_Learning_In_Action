import regTrees
from numpy import *

myDat = regTrees.loadDataSet('ex00.txt')
myMat = mat(myDat)
regTrees.createTree(myMat, ops = (0,1))
myDat2 = regTrees.loadDataSet('ex2.txt')
myMat2 = mat(myDat2)
print(regTrees.createTree(myMat2))
print(regTrees.createTree(myMat2, ops=(10000, 4)))
