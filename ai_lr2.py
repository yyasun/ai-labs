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
from datagen import *
from mathfuncs import *
from network import *

def printW(W):
    print("W:[")
    for w in W:
        print("[")
        for wj in w:
            print(" [ ", end="")
            for j in wj:
                print(j, end=", ")
            print("],")
        print("],")
    print("]",end="\n")

def forward(xs, W):
    #       input         , hidden , output
    net = [[x for x in xs], [0] * 3, [0,0]]
    for i, Wl in enumerate(W):
        for j, Wn in zip(range(len(net[i+1])),Wl):
            net[i+1][j] = sigmoid(np.dot(net[i], Wn))

    return net[2], net

def batchLoss(outpus, Y):
    sum = 0
    for o,y in zip(outpus, Y):
        sum += loss(o,y)
    return sum / len(outpus)

def backward(net, W, y, learningRate, loss):
    sigmas = [[0,0,0],[0,0]]
    for i, o in enumerate(net[-1]):
        sigmas[-1][i] = loss * dSigmoid(np.dot(net[-2],W[-1][i]))

    for i, sigma in enumerate(sigmas[0]):
        sigmas[0][i] = np.sum([s*W[1][j][i] for j, s in enumerate(sigmas[-1])]) * dSigmoid(np.dot(net[0],W[0][i]))
    
    for i, Wl in enumerate(W):
        for j, Wn in enumerate(Wl):
            for k, w in enumerate(Wn):
                W[i][j][k] -= learningRate * sigmas[i][j] * net[i][j]
    return

def train(X, Y, W, targetError, learningRate):
    for i in range(100):
        for xs, ys in zip(X,Y):
            backward(net, W, ys, learningRate, loss/len(xs))
        print( "╟──────────╫─────────────────────╫───────────────────")
        prevError = sumErr/20
    return W


# W = [[[random.random() for _ in range(3)] for _ in range(3)],
#     [[random.random() for _ in range(3)] for _ in range(2)]]  # Weights
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

net = Net(10)

generateData(func, X, Y, x1, x2, x3)

net.forward(X[0])


# print("X\tY")
# for xs, ys in zip(X,Y):
#     print(f"{xs}\t\t{ys}")

# printW(W)

# train(X, Y, W, target_error, 0.001)
# printW(W)
# print()
# sumErr = 0
# for xs, ys in zip(X,Y):
#     out, _ = forward(xs,W)
#     sumErr += out[0] - ys[0]
#     # out[1]=1 if out[1] > 0.6 else 0
#     print(f"{out}  \t{ys}")
# print()
# print(sumErr/len(X))

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