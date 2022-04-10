import math
import numpy as np

def ReLU(x): return max(0, x)  # ReLU activation function

def dReLU(x): return np.array([1 if i >= 0 else 0 for i in x])  # ReLU derivative function

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))  # Sigmoid activation function

def dSigmoid(x): return x * (1.0 - x)  # Sigmoid derivative function

def func(xs): return math.tan(xs[0]) + math.sin(xs[1]) - math.sin(xs[2])

def tlu(x): return 1 if x > 0.5 else 0

def loss(output, y):  return 1/2 * ((output[0] - y[0] + output[1] - y[1])/2)**2

def dLoss(output, y): return (output[0] - y[0] + output[1] - y[1])/2 