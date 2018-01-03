import trees

myDat, labels = trees.createDataSet()
print(myDat)
print(trees.calcShannonEnt(myDat))
myDat[0][-1] = 'maybe'
print(myDat)
print(trees.calcShannonEnt(myDat))
