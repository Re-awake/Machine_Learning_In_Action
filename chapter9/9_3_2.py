import regTrees
from numpy import *

myDat = regTrees.loadDataSet('ex00.txt')
myMat = mat(myDat)
print(regTrees.createTree(myMat))
myDat1 = regTrees.loadDataSet('ex0.txt')
myMat1 = mat(myDat1)
print(regTrees.createTree(myMat1))
