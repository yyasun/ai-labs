from ast import operator
from asyncio.windows_events import NULL
from distutils.log import error
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


def train(net: Net, iterations_count):
    error = 0
    prev_error = 0
    for iteration in range(iterations_count):
        deltas = NULL        
        for xs, ys in zip(X,Y):
            net.forward(xs)
            if deltas == NULL:
                deltas = net.backward(xs, ys)
            delta = net.backward(xs, ys)
            error += abs(net.output[0] - ys[0])
            for i in range(len(deltas)):
                deltas[i] += delta[i]
        error /= len(X)
        for i in range(len(deltas)):
                deltas[i] /= len(X)
        
        net.update_weights(deltas)
        print( "╟──────────╫─────────────────────╫───────────────────")
        print(f"{iteration+1}\t {error}\t {error-prev_error}")
        prev_error = error


set_size = 32
x1 = 9
x2 = 8
x3 = 7
target_error = 0.002

net = Net(learning_rate = 0.0001, hid_count = 10)

X, Y = generateData(func, set_size, x1, x2, x3)

train(net, 4000)

for xs, ys in zip(X,Y):
    net.forward(xs)    
    print(f"{net.output}  \t{ys}")



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

# def backward(net, W, y, learningRate, loss):
#     sigmas = [[0,0,0],[0,0]]

#     for i, o in enumerate(net[-1]):
#         sigmas[-1][i] = loss * dSigmoid(np.dot(net[-2],W[-1][i]))

#     for i, sigma in enumerate(sigmas[0]):
#         sigmas[0][i] = np.sum([s*W[1][j][i] for j, s in enumerate(sigmas[-1])]) * dSigmoid(np.dot(net[0],W[0][i]))
    
#     for i, Wl in enumerate(W):
#         for j, Wn in enumerate(Wl):
#             for k, w in enumerate(Wn):
#                 deltas[i][j][k] = sigmas[i][j] * net[i][j]
#     return