import regression

lgX = []
lgY = []
regression.setDataCollect(lgX, lgY)

print(shape(lgX))
lgX1 = mat(ones((58, 5)))
lgX1[:, 1:5] = mat(lgX)
print(lgX[0])
print(lgX1[0])

ws = regression.standRegres(lgX1, lgY)
print(ws)
print(lgX1[0] * ws)
print(lgX1[-1] * ws)
print(lgX1[43] * ws)
