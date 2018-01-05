from numpy import *
import adaboost

datMat, classLabels = adaboost.loadSimpData()
D = mat(ones((5, 1)) / 5)
adaboost.buildStump(datMat, classLabels, D)
