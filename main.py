import math
import random

# import gaussian as gaussian
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import gaussian
from scipy.stats import multivariate_normal


def g(x, var, miu):
    return 1 / (math.sqrt(var * 2 * math.pi)) * np.exp(-0.5 * np.power((x - miu), 2)/var)


def multi_dimensional_gaussian(variables,variance_list,miu_list):
    """
    X-list of decision variables
    m- Number of local optima
    ul-Interval span of side constraints
    r - Ratio of local optima to global optima(=1)
    w- w*ul is the variance
    d- all peaks must be in range [-d*ul/2,d*ul/2]
    miu- list of  an average point of an n dimensional Gaussian distribution function
    """

    index = 0
    g_result=1
    for x in variables:
        g_result *= multivariate_normal.pdf(x,variance_list[index],miu_list[index])
        index += 1
    return g_result

def G(m,Wi,variables,variance_list,miu_list):  #todo: set peaks in d range
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
        lst.append(Wi[i]*multi_dimensional_gaussian(variables,miu_list,variance_list))

    return lst[2] #not good

nList = [5, 10, 20, 40]
mList=[0, 5, 10, 20]

rList = [0.1, 0.3, 0.6, 0.9]
ulList=[10, 20, 30, 40]
wList = [0.01, 0.03, 0.06, 0.09]
dList = [0.25, 0.5, 0.75, 1.0]
Wlist= []
miu_list = []
for i in range(0,6):
    ra=1*random.random()
    miu_list.append(ra)
variance = [[] * 4 for i in range(4)]
for i in range(0, 4):
    for j in range(0,4):
        variance[i].append(np.random.uniform(0,ulList[i] * wList[j]))


#parameter choose:
n=2
m=mList[1]
ul=ulList[1]
r=rList[1]
w=wList[1]
d = dList[1]
for i in range(0,6):
    Wlist.append(random.random()*r)
#change-
var = []
var.append(variance[ulList.index(ul)][wList.index(w)])
var.append(variance[1][2])
variables = []
for i in range(0, n):
    variables.append(np.linspace(-ul / 2, ul / 2, 100))

#function
z=G(m,Wlist,variables,var,miu_list)

#plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot3D(variables[0],variables[1],z,'green')
plt.show()