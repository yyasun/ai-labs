# 13, 2, 12
from audioop import avg
from cmath import sqrt
import matplotlib.pyplot as plt
import numpy as np
import math
import random as rnd
from ast import literal_eval

class indiv:
    def __init__(self,x,y,fit):
        self.geneX = x
        self.geneY = y
        self.fit = fit



def ecly(x,y):
    res = -20 * math.exp(-0.2 * sqrt(0.5*(x**2 + y**2))) - math.exp(0.5*(math.cos(2*math.pi*x) + math.cos(2*math.pi*y))) + math.e + 20
    return res

def camel(x,y):
    return 2*x**2 - 1.05*x**4 + (x**6)/6 + x*y + y**2

def generate(count, fit):
    gen = []
    for i in range(count):
        x = rnd.uniform(-4,4)
        y = rnd.uniform(-4,4)        
        gen.append(indiv(x, y, fit(x,y)))
    return gen

mutation_chance = 0.3
cross_point = 3

i = 0
gen = generate(10, camel)

while i < 100:
    print(f"iteration: {i}")    
    print([g.fit for g in gen])
    # bread new gen
    avg_fit = sum([g.fit for g in gen])/len(gen)
    print(f"avg fit: {avg_fit}")
    # selection
    breaders = [indiv for indiv in gen if indiv.fit >= avg_fit]
    # started new gen
    for j in range(len(gen)):

    i+=1


# 1 - generate
# 2 - selection
# 3 - cross
# 4 - mutation
# 5 - repeat 1-4
