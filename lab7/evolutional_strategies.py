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

def mutate_z(subject: indiv, sigma, bounds):
    for gene_i in range(len(subject)):
        mutated = subject[gene_i] + np.random.normal(0, sigma)
        if mutated < bounds[gene_i][0]:
            mutated += bounds[gene_i][1] - bounds[gene_i][0]
        elif mutated > bounds[gene_i][1]:
            mutated += bounds[gene_i][0] - bounds[gene_i][1]
        subject[gene_i] = mutated

def mutate_s(gen: list):
    child = indiv([0]*len(gen[0]),0)
    for gene_i in range(len(child)):
        m = rnd.choice(gen)
        f = rnd.choice(gen)
        recomb = (m[gene_i] + f[gene_i])/2
    return child


def selection(gen: list, mu):
    sorted_gen = sorted(gen, key = lambda obj: obj.fit)
    return sorted_gen[:mu]

def deformed_stars(mu: int, lmbda: int, bounds: list, objective):
    gen = generate(mu, bounds, objective)
    best = 100000000
    sigma = 0.7
    for i in range(100):
        avg_fit = sum([g.fit for g in gen])/len(gen)
        gen_best = min([g.fit for g in gen]) 
        if best > gen_best:
            best = gen_best
        print(f"{i}. gen best fit: {gen_best}  |   avg fit: {avg_fit}\n")
    
        new_gen_z = []
        new_gen_s = []
        for j in range(lmbda):
            child_z = rnd.choice(gen)
            mutate_z(child_z, sigma, bounds)
            child_z.fit= objective(child_z[:])
            new_gen_z.append(child_z)
            child_s = mutate_s(gen)
            child_s.fit = objective(child_s[:])
            new_gen_s.append(child_s)
        new_gen = gen + new_gen_z + new_gen_s
        gen = selection(new_gen, mu)
    return best