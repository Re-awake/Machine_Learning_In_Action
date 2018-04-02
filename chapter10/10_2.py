import kMeans
from numpy import *

datMat = mat(kMeans.loadDataSet('testSet.txt'))
myCentroids, clustAssing = kMeans.kMeans(datMat, 4)
