from audioop import avg
from cmath import sqrt
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from ast import literal_eval
import collections

class indiv(list):
    def __init__(self, genes_v, fit_vlv):
        self.fit = fit_vlv
        self.extend(genes_v)
    
    def __str__(self):
        return f"{self[:]}, fit: {self.fit}"
    
    def __eq__(self, r):
        flag = True
        for i in range(len(r)):
            flag = flag and self[i] == r[i]
            if not flag:
                return False
        return flag

    def clone(self):
        return indiv(self[:], self.fit)



def generate(count, bounds, fit):
    gen = []
    for i in range(count):
        genes = list()
        for bound in bounds:
            genes.append(rnd.uniform(bound[0],bound[1]))
        gen.append(indiv(genes, fit(genes)))
    return gen

def crossover(m: indiv, f: indiv, fit):
    child1 = m.clone()
    child2 = f.clone()
    c_pnt = rnd.randint(1,len(f)-1)    
    child1[c_pnt:], child2[c_pnt:] = child2[c_pnt:], child1[c_pnt:]
    child1.fit = fit(child1[:])
    child2.fit = fit(child2[:])
    return [child1, child2]

def mutate(subject: indiv, fit, bounds, mutation_chance):
    for gene_i in range(len(subject)):
        if rnd.uniform(0,1) < mutation_chance:
            subject[gene_i] = np.clip(subject[gene_i] + rnd.uniform(-.5,.5),bounds[gene_i][0],bounds[gene_i][1])
    subject.fit = fit(subject[:])

def selection(gen: list):
    avg_fit = sum([g.fit for g in gen])/len(gen)
    breaders = [indiv for indiv in gen if indiv.fit <= avg_fit]
    return breaders

def GA(gen_len: int, mutation_chance: float, bounds: list, objective):
    mutation_chance = mutation_chance
    gen = generate(gen_len, bounds, objective)
    best = 100000000

    for i in range(1000):
        avg_fit = sum([g.fit for g in gen])/len(gen)
        gen_best = min([g.fit for g in gen]) 
        if best > gen_best:
            best = gen_best
        print(f"{i}. gen best fit: {gen_best}  |   avg fit: {avg_fit}\n")
    
        breaders = selection(gen)
        if not breaders or len(breaders) == 1:
            breaders = gen
        new_gen = []
        for j in range(int(len(gen)/2)):
            mother = rnd.choice(breaders)
            father = rnd.choice(breaders)
            while mother == father: # select two different parents :)
                mother = rnd.choice(breaders)
                father = rnd.choice(breaders)    
            children = crossover(mother, father, objective)
            [mutate(child, objective, bounds, mutation_chance) for child in children]
            new_gen.extend(children)
        gen = new_gen
    return best
