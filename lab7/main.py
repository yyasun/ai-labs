# 13, 2, 12
import evolutional_strategies as ES
import math
from math import cos, exp, pi, sqrt, e, log


def func(v):
    x = v[0]
    return (100 * sqrt(100-x**2)*cos(x**2)*cos(x))/((x**2+10)*log(100-x**2))

bounds = [[-5,5]]
best = ES.deformed_stars(10, 40, bounds, func)

print(f"best: {best}")