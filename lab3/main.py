# 13, 2, 12
import genetic 
import math

def ecly(v: list):
    x, y = v
    res = -20 * math.exp(-0.2 * math.sqrt(0.5*(x**2 + y**2))) - math.exp(0.5*(math.cos(2*math.pi*x) + math.cos(2*math.pi*y))) + math.e + 20
    return res

def camel(v: list):
    x, y = v
    return 2*x**2 - 1.05*x**4 + (x**6)/6 + x*y + y**2

mutation_chance = 0.1
bounds = [[-4,4],[-4,4]]
best = genetic.GA(100,mutation_chance, bounds, camel)

print(f"best: {best}")
# print(f"best fit: {min([g.fit for g in gen])}")


# 1 - generate
# 2 - selection
# 3 - cross
# 4 - mutation
# 5 - repeat 1-4
