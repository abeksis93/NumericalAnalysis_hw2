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
    return gResult

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
    lst = []
    for i in range(0,m+1):
        lst.append(Wi*g(X,m,ul,r,w,d,miu))

    return max(lst)

nList=[5, 10, 20, 40]
mList=[0, 5, 10, 20]
ulList=[10, 20, 30, 40]
rList = [0.1, 0.3, 0.6, 0.9]
w = [0.01, 0.03, 0.06, 0.09]
d = [0.25, 0.5, 0.75, 1.0]
Gaussians=[]
for n in nList:
    Gaussians.append(G(n,mList[1],ulList[1],rList1[1],wList[1],dList[1]))
