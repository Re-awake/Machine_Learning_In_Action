import regression
import matplotlib.pyplot as plt
from numpy import *

xArr, yArr = regression.loadDataSet('ex0.txt')
print(yArr[0])
print(regression.lwlr(xArr[0], xArr, yArr, 1.0))
print(regression.lwlr(xArr[0], xArr, yArr, 0.001))

# Test with k = 0.003, cause overfitting
yHat = regression.lwlrTest(xArr, xArr, yArr, 0.003)
xMat = mat(xArr)
srtInd = xMat[:, 1].argsort(0)
xSort = xMat[srtInd][:, 0, :]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:, 1], yHat[srtInd])
ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s = 2, \
            c = 'red')
plt.show()
