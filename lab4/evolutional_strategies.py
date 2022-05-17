from audioop import avg
from cmath import sqrt
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from ast import literal_eval
import collections

step_size_control = 0.2
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
    
    def __ln__(self, r):
        return self.fit < r.fit
    
    def __cmp__(self, r):
        return self.fit.__cmp__(r.fit)

def generate(count, bounds, fit):
    gen = []
    for i in range(count):
        genes = list()
        for bound in bounds:
            genes.append(rnd.uniform(bound[0],bound[1]))
        gen.append(indiv(genes, fit(genes)))
    return gen

def recombine(gen: list, bounds): # global intermideate
    child = indiv([0]*len(gen[0]),0)
    for gene_i in range(len(child)):
        m = rnd.choice(gen)
        f = rnd.choice(gen)
        recomb = (m[gene_i] + f[gene_i])/2
        child[gene_i] = np.clip(recomb, bounds[gene_i][0],bounds[gene_i][1])
    return child

def mutate(subject: indiv, sigma, bounds):
    for gene_i in range(len(subject)):
        mutated = subject[gene_i] + np.random.normal(0, sigma)
        subject[gene_i] = np.clip(mutated, bounds[gene_i][0],bounds[gene_i][1])
    # sigma += np.random.normal(0, step_size_control)

def selection(gen: list, mu):
    sorted_gen = sorted(gen, key = lambda obj: obj.fit)
    return sorted_gen[:mu]

def comma(mu: int, lmbda: int, bounds: list, objective):
    gen = generate(mu, bounds, objective)
    best = 100000000
    sigma = 0.7
    for i in range(10):
        avg_fit = sum([g.fit for g in gen])/len(gen)
        gen_best = min([g.fit for g in gen]) 
        if best > gen_best:
            best = gen_best
        print(f"{i}. gen best fit: {gen_best}  |   avg fit: {avg_fit}\n")
    
        new_gen = []
        for j in range(lmbda): 
            child = recombine(gen, bounds)
            mutate(child, sigma, bounds)
            child.fit= objective(child[:])
            new_gen.append(child)
        gen = selection(new_gen, mu)
    return best
