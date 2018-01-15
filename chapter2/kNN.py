from numpy import *
from os import listdir
import operator

# Create data for later usage, chapter 2_1_1
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# kNN algorithm, program 2_1
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # Distance between two points, using eculidean
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    # Finding the k point of shortest distance
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # Sorting
    sortedClassCount = sorted(classCount.items(),
      key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

# Read txt file into matrix, program 2_2
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    # Get the lines of file
    numberOfLines = len(arrayOLines)
    # Create a matrix with all zeros
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # Read numbers into matrix
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# Normalization of features, program 2_3
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # Normalization Step, dataset minus minimum value
    # tile function means create a matrix with m row, repeating minVals for
    # column once.
    normDataSet = dataSet - tile(minVals, (m, 1))
    # Normalization Step, divide the range of value
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# Classifer for dating website, testing its accuracy, program 2_4
def datingClassTest():
    # Ratio of testing case
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    # Testing classifier
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                            datingLabels[numTestVecs:m], 3)
        print("The classifier came back with: %d, the real answer is: %d" \
                % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i] :
            errorCount += 1.0
    print("The total error rate is: %f" %(errorCount/float(numTestVecs)))

# Dating Website Prediction Function, program 2_5
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input(
                    "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - \
                        minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", \
                    resultList[classifierResult - 1])

# Change image of 32 x 32 pixels to 1 x 1024 vector, chapter 2_3_1
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])
    return returnVect

# Handwriting Testing Function, chapter 2_6
def handwritingClassTest():
    # Retrieve directory content
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    # Analysis the number from the file name, training the data
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    # Testing the algorithm
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('_')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, \
                                trainingMat, hwLabels, 3)
        print("The classifier came back with: %d, the real answer is: %d"\
                % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))
