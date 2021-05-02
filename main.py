import math

import gaussian as gaussian
import matplotlib as plt
import numpy as np


def gaussian(x, var, miu):
    return 1 / (math.sqrt(var * 2 * math.pi)) * np.exp(-0.5 * np.power((x - miu), 2)/var)


def g(X, m, ul, r, w, d, miu):
    index = 0
    gResult = 1
    for x in X:
        gResult *= r*gaussian(x, w*ul, miu[index])
        index += 1


