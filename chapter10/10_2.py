import kMeans
import os
import sys
from numpy import *

project_path = os.path.abspath(os.path.dirname(__file__))
text_path = os.path.join(project_path, "../chapter10/testSet.txt")
datMat = mat(kMeans.loadDataSet(text_path))
myCentroids, clustAssing = kMeans.kMeans(datMat, 4)
