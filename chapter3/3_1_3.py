import trees

myDat, labels = trees.createDataSet()
myTree = trees.createTree(myDat, labels)
print(myTree)
