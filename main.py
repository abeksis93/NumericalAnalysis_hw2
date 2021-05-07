import math
import random

# import gaussian as gaussian
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import gaussian
from scipy.stats import multivariate_normal
from mpl_toolkits import mplot3d




def g(x, var, miu):
    return 1 / (math.sqrt(var * 2 * math.pi)) * np.exp(-0.5 * np.power((x - miu), 2)/var)


def multivar_gaussian(variables,variance_list,miu_list):
    """
    X-list of decision variables
    m- Number of local optima
    ul-Interval span of side constraints
    r - Ratio of local optima to global optima(=1)
    w- w*ul is the variance
    d- all peaks must be in range [-d*ul/2,d*ul/2]
    miu- list of  an average point of an n dimensional Gaussian distribution function
    """

    matrix = [[]*len(variables[0])]*len(variables[0])

    for i in range(0,len(variables[0])):
        for j in range(0,len(variables[0])):
            g_x1 = multivariate_normal(variables[0][i], variance_list[0], miu_list[0])
            g_x2 = multivariate_normal(variables[0][j], variance_list[0], miu_list[0])
            matrix[i].append(g_x1.abseps*g_x2.abseps)


    return matrix

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
    g_lst=[]
    for g in range(m+1):
        for i in range(0, len(variables[0])):
            for j in range(0, len(variables[0])):
                g_lst.append(multivar_gaussian(variables,variance_list,miu_list))
                g_lst[g][i][j] = g_lst[g][i][j]*Wi[g];



    max_matrix = [[]*len(variables[0])]*len(variables[0])
    valueList=[]
    for i in range(0, len(variables[0])):
        for j in range(0, len(variables[0])):
            for g in range(0,m+1):
                valueList.append(g_lst[g][i][j])

        max_matrix[i].append(max(valueList))

    return max_matrix

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

var = []
var.append(variance[ulList.index(ul)][wList.index(w)])
var.append(variance[1][2])
variables = []
for i in range(0, n):
    variables.append(np.linspace(-ul / 2, ul / 2, 200))

#function
z=G(m,Wlist,variables,var,miu_list)

#plot
fig = plt.figure()

x,y=np.meshgrid(variables[0],variables[1])

zArray = np.array(z)

ax =plt.axes(projection='3d')
ax.plot_surface(x,y,zArray)
plt.show()