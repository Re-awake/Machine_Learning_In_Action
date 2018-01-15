# Port the graph using normalized data, chapter 2_2_3
import kNN

datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
print(normMat)
print(ranges)
print(minVals)
