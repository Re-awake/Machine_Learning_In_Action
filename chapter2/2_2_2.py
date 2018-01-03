import matplotlib, kNN
import matplotlib.pyplot as plt
from numpy import *

datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels),15.0*array(datingLabels))
plt.show()
