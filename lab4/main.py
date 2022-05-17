# 13, 2, 12
import evolutional_strategies as ES
import math
from math import cos, exp, pi


def izum(v: list):
    x, y = v
    return - cos(x)*cos(y)*exp(-((x - pi)**2+(y - pi)**2))

bounds = [[-5,5],[-5,5]]
best = ES.comma(10, 40, bounds, izum)

print(f"best: {best}")
# print(f"best fit: {min([g.fit for g in gen])}")


# 1 - generate
# 2 - selection
# 3 - cross
# 4 - mutation
# 5 - repeat 1-4
