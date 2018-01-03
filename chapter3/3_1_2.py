import trees

a = [1, 2, 3]
b = [4, 5, 6]
a.append(b)
print(a)

a = [1, 2, 3]
a.extend(b)
print(a)

myDat, labels = trees.createDataSet()
print(myDat)
print(trees.splitDataSet(myDat, 0, 1))
print(trees.splitDataSet(myDat, 0, 0))

myDat, labels = trees.createDataSet()
print(trees.chooseBestFeatureToSplit(myDat))
print(myDat)
