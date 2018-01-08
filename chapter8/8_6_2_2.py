import regression

lgX = []
lgY = []
regression.setDataCollect(lgX, lgY)

lgX1 = mat(ones((58, 5)))
lgX1[:, 1:5] = mat(lgX)

regression.corssValidation(lgX, lgY, 10)
regression.redgeTest(lgX, lgY)
