from numpy import *
import adaboost

datMat, classLabels = adaboost.loadSimpData()
classifierArray = adaboost.adaBoostTrainDS(datMat, classLabels, 9)
print(classifierArray)
