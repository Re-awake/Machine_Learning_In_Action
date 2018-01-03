# import numpy library
from numpy import *
# randomly generate four lists of four elements
print("Matrix is:\n", random.rand(4, 4))
# change the list into a 4 by 4 matrix
randMat = mat(random.rand(4, 4))
# calculate the reverse of the matrix
invRandMat = randMat.I
print("Inverse is:\n", invRandMat.I)
# Matrix * Inverse = Identity Matrix
myEye = randMat * invRandMat
print("Matrix * Inverse is:\n", myEye)
# slight difference with real Identity Matrix deal to computation error
diff = myEye - eye(4)
print("Difference is:\n", diff)
