import regTrees
from numpy import *

myMat2 = mat(regTrees.loadDataSet('exp2.txt'))
print(regTrees.createTree(myMat2, regTrees.modelLeaf, regTrees.modelErr, \
    (1, 10)))
