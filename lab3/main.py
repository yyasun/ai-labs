# 13, 2, 12
from audioop import avg
from cmath import sqrt
import matplotlib.pyplot as plt
import numpy as np
import math
import random as rnd
from ast import literal_eval

class indiv(list):
    def __init__(self, genes_v, fit_f):
        self.fit = fit_f(genes_v)
        self.extend(genes_v)
    
    def __str__(self):
        return f"{self[:]}, fit: {self.fit}"

    def clone(self):
        return indiv(self[:], self.fit)

def ecly(v: list):
    x, y = v
    res = -20 * math.exp(-0.2 * sqrt(0.5*(x**2 + y**2))) - math.exp(0.5*(math.cos(2*math.pi*x) + math.cos(2*math.pi*y))) + math.e + 20
    return res

def camel(v: list):
    x, y = v
    return 2*x**2 - 1.05*x**4 + (x**6)/6 + x*y + y**2

def generate(count, fit):
    gen = []
    for i in range(count):
        x = rnd.uniform(-4,4)
        y = rnd.uniform(-4,4)        
        gen.append(indiv([x, y], fit))
    return gen

def crossover(m: indiv, f: indiv, fit):
    child1 = m.clone()
    child2 = f.clone()
    c_pnt = rnd.randint(1,len(f)-1)    
    child1[c_pnt:], child2[c_pnt:] = child2[c_pnt:], child1[c_pnt:]
    child1.fit = fit(child1[:])
    child2.fit = fit(child2[:])
    return [child1, child2]

def mutate(i: indiv, fit):
    if rnd.uniform(0,1) > mutation_chance:
        return
    multipe = rnd.randint(-10,2)
    mut_value  = 2**multipe if rnd.uniform(0,1) > 0.5 else -(2**multipe)
    if rnd.uniform(0,1) > 0.5:
        i.geneX = np.clip(i.geneX + rnd.gauss(-0.5,0.5),-4,4)
    if rnd.uniform(0,1) > 0.5:
        i.geneY = np.clip(i.geneY + rnd.gauss(-0.5,0.5),-4,4)
    i.fit = fit(i.geneX, i.geneY)

def selection(gen: list):
    avg_fit = sum([g.fit for g in gen])/len(gen)
    breaders = [indiv for indiv in gen if indiv.fit <= avg_fit]
    if len(breaders) == 0:
        breaders = gen
    return breaders

mutation_chance = 0.1
bounds = [[-4,4],[-4,4]]
objective = camel
gen = generate(10, objective)

for i in range(1000):
    # print([g.fit for g in gen])
    avg_fit = sum([g.fit for g in gen])/len(gen)
    # print(f"{i}. best fit: {min([g.fit for g in gen])}    |   avg fit: {avg_fit}\n")
    
    breaders = selection(gen)
    
    new_gen = []
    # started new gen
    for j in range(len(gen)):
        mother = rnd.choice(breaders)
        father = rnd.choice(breaders)
        # crossover
        child = crossover(mother, father, objective)
        # mutation
        mutate(child, objective)
        new_gen.append(child)
    gen = new_gen

print(f"best fit: {min([g.fit for g in gen])}")


# 1 - generate
# 2 - selection
# 3 - cross
# 4 - mutation
# 5 - repeat 1-4
