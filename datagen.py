import random

def normalize(X,Y):
    for i,xs in enumerate(X):       
        x1 = [xs[0] for xs in X]
        x2 = [xs[1] for xs in X]
        x3 = [xs[2] for xs in X]
        maxX = [max(x1),max(x2), max(x3)]
        minX = [min(x1),min(x2), min(x3)]
        for j,x in enumerate(xs):    
            X[i][j] = (x - minX[j])/(maxX[j] - minX[j])

    minY = min([ys[0] for ys  in Y])
    maxY = max([ys[0] for ys  in Y])

    minY1 = min([ys[1] for ys  in Y])
    maxY1 = max([ys[1] for ys  in Y])
    for i, ys in enumerate(Y):
        Y[i][0] = (ys[0] - minY)/(maxY - minY)
        Y[i][1] = (ys[1] - minY1)/(maxY1 - minY1)

def generateData(func, set_size, x1, x2, x3):
    X = [0]*set_size
    Y = [0]*set_size
    for i in range(set_size):
        X[i] = [0] * 3
        Y[i] = [0] * 2

    for i in range(len(X)):
        for j in range(len(X[i])):
            if i == 0 and j == 0:
                X[i][j] = x1
                continue
            if i == 0 and j == 1:
                X[i][j] = x2
                continue
            if i == 0 and j == 2:
                X[i][j] = x3
                continue
            else:
                r = random.random()
                if r > 0.5:
                    X[i][j] = X[i - 1][j] + 1
                else:
                    X[i][j] = X[i - 1][j] - 1

    sum = 0
    for i in range(len(Y)):
        res = func(X[i])
        Y[i][0] = res  
        sum += res

    avg = sum / 20

    for i in range(len(Y)):
        for j in range(len(Y[i])):
            if j == 1:
                if func(X[i]) > avg:
                    Y[i][j] = 1
    
    normalize(X, Y)

    return X, Y