import regTrees
from numpy import *

myDat = regTrees.loadDataSet('ex00.txt')
myMat = mat(myDat)
myDat2 = regTrees.loadDataSet('ex2.txt')
myMat2 = mat(myDat2)

myTree = regTrees.createTree(myMat2, ops=(0, 1))
myDatTest = regTrees.loadDataSet('ex2test.txt')
myMat2Test = mat(myDatTest)
regTrees.prune(myTree, myMat2Test)
