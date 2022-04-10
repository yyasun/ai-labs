import numpy as np
from mathfuncs import *

class Net:
    def __init__(self, learning_rate, hid_count, in_count = 3, out_count= 2):
        self.input = np.zeros(in_count)
        self.hidden = np.zeros(hid_count)
        self.output = np.zeros(out_count)
        self.learning_rate = learning_rate
        
        self.neurons = [self.input,self.hidden, self.output]
        self.weights = [
            np.random.rand(hid_count, in_count),
            np.random.rand(out_count, hid_count )]        


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
            for j, neuron_weights in enumerate(Wl):
                activation = np.dot(self.neurons[i], neuron_weights)
                self.neurons[i+1][j] = sigmoid(activation)
            print()


    def backward(self, xs, ys):
        deltas = [
            np.zeros(self.weights[0].shape),
            np.zeros(self.weights[1].shape)]
        return deltas


    def update_weights(self, deltas: list[np.ndarray]):
        for i in range(self.weights):
            self.weights[i] -= self.learning_rate * deltas[i]