# 13, 2, 12
from audioop import avg
from cmath import sqrt
import matplotlib.pyplot as plt
import numpy as np
import math
import random as rnd
from ast import literal_eval

class indiv:
    def __init__(self,x = 0.0,y = 0.0,fit = 0.0):
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

def crossover(m: indiv, f: indiv, fit) -> indiv:
    child = indiv()
    if rnd.uniform(0,1) < 0.5:
        child.geneX = f.geneX
        child.geneY = m.geneY
    else:
        child.geneX = m.geneY 
        child.geneY = f.geneX
    child.fit = fit(child.geneX, child.geneY)
    return child

def mutate(i: indiv, fit):
    if rnd.uniform(0,1) > mutation_chance:
        return
    multipe = rnd.randint(-10,2)
    mut_value  = 2**multipe if rnd.uniform(0,1) > 0.5 else -(2**multipe)
    if rnd.uniform(0,1) > 0.5:
        i.geneX = np.clip(i.geneX + mut_value,-4,4)
    else:
        i.geneY = np.clip(i.geneY + mut_value,-4,4)
    i.fit = fit(i.geneX, i.geneY)

mutation_chance = 0.1

gen = generate(10, camel)

for i in range(10000):
    # print(f"iteration: {i}")    
    # print([g.fit for g in gen])
    avg_fit = sum([g.fit for g in gen])/len(gen)
    # print(f"best fit: {min([g.fit for g in gen])}    |   avg fit: {avg_fit}")
    # print()
    # bread new gen
    
    
    # selection
    breaders = [indiv for indiv in gen if indiv.fit <= avg_fit]
    if len(breaders) == 0:
        breaders = gen
    new_gen = []
    # started new gen
    for j in range(len(gen)):
        mother = rnd.choice(breaders)
        father = rnd.choice(breaders)
        # crossover
        child = crossover(mother, father, camel)
        # mutation
        mutate(child, camel)
        new_gen.append(child)
    gen = new_gen

print(f"best fit: {min([g.fit for g in gen])}")


# 1 - generate
# 2 - selection
# 3 - cross
# 4 - mutation
# 5 - repeat 1-4
