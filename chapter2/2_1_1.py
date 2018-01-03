import kNN

if __name__ == "__main__":
    group, labels = kNN.createDataSet()
    print(group)
    print(labels)
    result = kNN.classify0([0, 0], group, labels, 3)
    print(result)
    datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
    print(datingDataMat)
    print(datingLabels[0:20])
