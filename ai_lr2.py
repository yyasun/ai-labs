from ast import operator
from asyncio.windows_events import NULL
import enum
from gettext import NullTranslations
import math
import random
from functools import reduce
from re import S
import numpy as np
import matplotlib.pyplot as plt

def ReLU(x): return max(0, x)  # ReLU activation function

def dReLU(x): return np.array([1 if i >= 0 else 0 for i in x])  # ReLU derivative function

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))  # Sigmoid activation function

def dSigmoid(x): return x * (1.0 - x)  # Sigmoid derivative function

def func(xs): return math.tan(xs[0]) + math.sin(xs[1]) - math.sin(xs[2])

def tlu(x): return 1 if x > 0.5 else 0

def loss(output, y):  return 1/2 * ((output[0] - y[0] + output[1] - y[1])/2)**2

def dLoss(output, y): return (output[0] - y[0] + output[1] - y[1])/2 

def batchLoss(outpus, Y):
    sum = 0
    for o,y in zip(outpus, Y):
        sum += loss(o,y)
    return sum / len(outpus)

def printW(W):
    print("W:")
    for w in W:
        print("[")
        for wj in w:
            print(" [ ", end="")
            for j in wj:
                print(j, end=" ")
            print("]")
        print("]")
    print()

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

def generateData(func, X, Y, x1, x2, x3):
    for i in range(20):
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

def infer(xs, W):
    #       input         , hidden , output
    net = [[x for x in xs], [0] * 3, [0,0]]
    for i, Wl in enumerate(W):
        for j, Wn in zip(range(len(net[i+1])),Wl):
            net[i+1][j] = sigmoid(np.dot(net[i], Wn))

    return net[2], net

def adjustWeights(net, W, y, learningRate):
    sigmas = [[0,0,0],[0,0]]
    for i, o in enumerate(net[-1]):
        sigmas[-1][i] = dLoss(net[-1],y) * dSigmoid(np.dot(net[-2],W[-1][i]))

    for i, sigma in enumerate(sigmas[0]):
        sigmas[0][i] = np.sum([s*W[1][j][i] for j, s in enumerate(sigmas[-1])]) * dSigmoid(np.dot(net[0],W[0][i]))
    
    for i, Wl in enumerate(W):
        for j, Wn in enumerate(Wl):
            for k, w in enumerate(Wn):
                W[i][j][k] += learningRate * sigmas[i][j] * net[i][j]
    return

def train(X, Y, W, targetError, learningRate):
    outputs = []
    batch = [xy for xy in zip(X,Y)]
    error = 1e-6
    i = 1
    print()
    print( "╔══════════╦═════════════╦════════════════════════╗")
    print(f"║batch\t   ║loss\t ║")
    # while error > targetError:
    prevError = 0 
    for i in range(1000000):
        # random.shuffle(batch)
        sumErr=0
        net = NULL
        outputs = []
        for xs, ys in batch:
            output, net = infer(xs, W)
            outputs.append(output)
            sumErr += abs(output[0] - ys[0])
            adjustWeights(net, W, ys, learningRate)
        loss = batchLoss(outputs, [b[1] for b in batch])
        print( "╟──────────╫─────────────────────╫───────────────────")
        
        print(f"║{i+1}\t   ║{sumErr/20}\t ║{sumErr/20-prevError}")
        prevError=sumErr/20
        # printW(W)
        i += 1
    print("╚══════════╩═════════════════════╝")
    return W


W = [[[random.random() for _ in range(3)] for _ in range(3)],
    [[random.random() for _ in range(3)] for _ in range(2)]]  # Weights
W = [[
 [ 0.7567664551478331, 0.4136089360015768, 0.003546293406428788 ],
 [ 0.12923728865066778, 0.8979589619102344, 0.04732918691784152 ],
 [ 0.4043120554459646, 0.4410567959324414, 0.36266243849402335 ]
],
[
 [ 1.1029705491918562, 1.1398820245076227, 1.3452342850773622 ],
 [ 0.15658532078355336, -0.04310559632456144, 0.43894343409792724 ]
]]
X = [0] * 20
Y = [0] * 20
x1 = 9
x2 = 8
x3 = 7
target_error = 0.002

generateData(func, X, Y, x1, x2, x3)

print(f"X:\n{X}",end="\n\n")
print(f"Y:\n{Y}",end="\n\n")
printW(W)

# train(X, Y, W, target_error, 0.00001)
printW(W)

sumErr = 0
for xs, ys in zip(X,Y):
    out, _ = infer(xs,W)
    sumErr += out[0] - ys[0]
    # out[1]=1 if out[1] > 0.6 else 0
    print(f"{out}  \t{ys}")
print()
print(sumErr/len(X))

"""
[
 [ 0.7920759689014302 0.44891844975519657 0.03885580716004525 ]
 [ 0.12476020569622498 0.8934818789558124 0.04285210396339716 ]
 [ 0.6089709981271537 0.6457157386136275 0.5673213811752207 ]
]
[
 [ 0.5824740094831689 0.6193854847989396 0.824737745368701 ]
 [ 0.2234546043001241 0.02376368719200861 0.5058127176144986 ]
]
"""