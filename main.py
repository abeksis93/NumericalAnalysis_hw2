import math
from random import random

import gaussian as gaussian
import matplotlib.pl as plt
import numpy as np


def gaussian(x, var, miu):
    return 1 / (math.sqrt(var * 2 * math.pi)) * np.exp(-0.5 * np.power((x - miu), 2)/var)


def g(X,m,ul,r,w,d,miu):
    """
    X-list of decision variables
    m- Number of local optima
    ul-Interval span of side constraints
    r - Ratio of local optima to global optima(=1)
    w- w*ul is the variance
    d- all peaks must be in range [-d*ul/2,d*ul/2]

    """
    index=0
    gResult=1
    for x in X:
        gResult *= gaussian(x,w*ul,miu[index])
        index+=1

def G(X,m,ul,r,w,d,miu,Wi):
    """
    X-list of decision variables
    m- Number of local optima
    ul-Interval span of side constraints
    r - Ratio of local optima to global optima(=1)
    w- w*ul is the variance
    d- all peaks must be in range [-d*ul/2,d*ul/2]
    miu- list of  an average point of an n dimensional Gaussian distribution function
    Wi - list of m+1 weights for each g_i
    """



