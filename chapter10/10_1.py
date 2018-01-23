import kMeans
from numpy import *

datMat = mat(kMeans.loadDataSet('testSet.txt'))
print(min(datMat[:, 0]))
print(min(datMat[:, 1]))
print(max(datMat[:, 1]))
print(max(datMat[:, 0]))

print(kMeans.randCent(datMat, 2))
