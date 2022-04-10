import numpy as np
from mathfuncs import *

class Net:
    def __init__(self, hid_count, in_count = 3, out_count= 2):
        self.input = np.zeros(in_count)
        self.hidden = np.zeros(hid_count)
        self.output = np.zeros(out_count)
        
        
        self.neurons = [self.input,self.hidden, self.output]
        self.weights = [
            np.random.rand(hid_count, in_count),
            np.random.rand(out_count, hid_count )]          
        print(self.weights[0].shape)
        print(self.weights[1].shape)

    def printW(self):
        print("W:[")
        for w in self.weights:
            print("[")
            for wj in w:
                print(" [ ", end="")
                for j in wj:
                    print(j, end=", ")
                print("],")
            print("],")
        print("]",end="\n")


    def forward(self, xs):
        self.neurons[0] = xs
        for i, Wl in enumerate(self.weights):
            for j, Wn in zip(range(len(self.neurons[i+1])),Wl):
                print(Wn)
                print(self.neurons[i])
                self.neurons[i+1][j] = sigmoid(np.dot(self.neurons[i], Wn))
            print()

    def backward(self, xs, ys):
        deltas = [
            np.zeros(self.weights[0].shape),
            np.zeros(self.weights[1].shape)]
        return deltas