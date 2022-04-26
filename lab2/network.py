from tkinter import E
import numpy as np
from mathfuncs import *

class Net:
    def __init__(self, learning_rate, hid_count, in_count = 3, out_count= 2):
        self.learning_rate = learning_rate

        self.input  =  np.zeros(in_count)
        self.hidden = np.zeros(hid_count)
        self.output = np.zeros(out_count)
        
        
        self.neurons = [self.input, self.hidden, self.output]
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
        self.neurons[0] = np.array(xs)
        for i, Wl in enumerate(self.weights):
            for j, neuron_weights in enumerate(Wl):
                activation = np.dot(self.neurons[i], neuron_weights)
                self.neurons[i+1][j] = sigmoid(activation)

        output = self.neurons[-1]

    def backward(self, xs, ys):
        deltas = [
            np.zeros(self.weights[0].shape),
            np.zeros(self.weights[1].shape)]
        sigmas = [
            np.zeros(self.neurons[1].shape),
            np.zeros(self.neurons[-1].shape)]

        # calc sigmas for output layer
        for i, o in enumerate(self.neurons[-1]):
            activation = np.dot(self.neurons[-2], self.weights[-1][i]) # Z * W
            sigmas[-1][i] = dLoss(self.neurons[-1], ys) * dSigmoid(activation) # dJ * dy

        #calc sigmas for hidden layer
        for i, sigma in enumerate(sigmas[0]):
            activation = np.dot(self.neurons[0], self.weights[0][i])
            sigmas[0][i] = np.sum([s * self.weights[1][j][i] 
                for j, s in enumerate(sigmas[-1])]) * dSigmoid(activation)
        
        for i, W_layer in enumerate(self.weights):
            for j, W_neuron in enumerate(W_layer):
                for k in range(W_neuron.size):                    
                    deltas[i][j][k] = sigmas[i][j] * self.neurons[i][k]    

        return deltas


    def update_weights(self, deltas: list[np.ndarray]):
        for i in range(len(self.weights)):
            self.weights[i] -=- self.learning_rate * deltas[i]