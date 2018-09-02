import kMeans
import os
import sys
from numpy import *

project_path = os.path.abspath(os.path.dirname(__file__))
text_path = os.path.join(project_path, "../chapter10/testSet.txt")
datMat = mat(kMeans.loadDataSet(text_path))
print(min(datMat[:, 0]))
print(min(datMat[:, 1]))
print(max(datMat[:, 1]))
print(max(datMat[:, 0]))

print(kMeans.randCent(datMat, 2))

print(kMeans.distEclud(datMat[0], datMat[1]))
