import adaboost

datArr, labelArr = adaboost.loadSimpData()
classifierArr = adaboost.adaBoostTrainDS(datArr, labelArr, 30)

adaboost.adaClassify([0, 0], classifierArr)
adaboost.adaClassify([[5, 5], [0, 0]], classifierArr)
