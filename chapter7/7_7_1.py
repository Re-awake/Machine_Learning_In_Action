import adaboost

datArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
classifierArray, aggClassEst = adaboost.adaBoostTrainDS(datArr, labelArr, 10)
adaboost.plotROC(aggClassEst.T, labelArr)
