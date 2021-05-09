import math
import random

# import gaussian as gaussian
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def g(x, var, miu):
    return 1 / (math.sqrt(var * 2 * math.pi)) * np.exp(-0.5 * np.power((x - miu), 2) / var)


def multivar_gaussian(variables, variance_list, miu_list):
    """
    X-list of decision variables
    m- Number of local optima
    ul-Interval span of side constraints
    r - Ratio of local optima to global optima(=1)
    w- w*ul is the variance
    d- all peaks must be in range [-d*ul/2,d*ul/2]
    miu- list of  an average point of an n dimensional Gaussian distribution function
    """

    matrix = []
    col = []

    for i in range(0, len(variables[0])):
        for j in range(0, len(variables[0])):
            g_x1 = g(variables[0][i], variance_list[0], miu_list[0])
            g_x2 = g(variables[1][j], variance_list[1], miu_list[1])
            col.append(g_x1 * g_x2)
        matrix.append(col)

    return matrix


def G(m, Wi, variables, variance_list, miu_list):  # todo: set peaks in d range
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
    g_lst = []
    max_val = 0
    for i in range(m + 1):
        g_lst.append(multivariate_normal(miu_list[i], np.diag(variance_list[i])))
        # for i in range(0, len(variables[0])):
        #     for j in range(0, len(variables[0])):
        # g_lst.append(multivar_gaussian(variables,variance_list,miu_list))
        # g_lst[g][i][j] = g_lst[g][i][j]*Wi[g];
        if max_val < g_lst[i].pdf(miu_list[i]):
            max_val = g_lst[i].pdf(miu_list[i])
            f = g_lst[i]

    # max_matrix = []
    # col=[]
    # valueList=[]
    # for i in range(0, len(variables[0])):
    #     for j in range(0, len(variables[0])):
    #         for g in range(0,m+1):
    #             valueList.append(g_lst[g][i][j])
    #
    #         col.append(max(valueList))
    #     max_matrix.append(col)

    return lambda x: sum(
        r * (1 - 0.05 * i) * g_lst[i].pdf(x) if g_lst[i] != f else f.pdf(x) / max_val for i in range(m + 1))


nList = [5, 10, 20, 40]
mList = [0, 5, 10, 20]

rList = [0.1, 0.3, 0.6, 0.9]
ulList = [10, 20, 30, 40]
wList = [0.01, 0.03, 0.06, 0.09]
dList = [0.25, 0.5, 0.75, 1.0]
Wlist = []
miu_list = []

# parameter choose:
n = 2
m = mList[1]
ul = ulList[1]
r = rList[1]
w = wList[1]
d = dList[1]
for i in range(0, 6):
    Wlist.append(random.random() * r)

variance_list = []
for i in range(0, m + 1):
    miu_list.append([random.random(), random.random()])
    variance_list.append([np.random.uniform(0, ulList[random.randint(0, 3)] * wList[random.randint(0, 3)]),
                          np.random.uniform(0, ulList[random.randint(0, 3)] * wList[random.randint(0, 3)])])

variables = []
for i in range(0, n):
    variables.append(np.linspace(-ul / 2, ul / 2, 500))

# function
z = G(m, Wlist, variables, variance_list, miu_list)

# plot
fig = plt.figure()

x, y = np.meshgrid(variables[0], variables[1])
# pos = np.empty(x.shape + (2,))
# pos[:, :, 0] = x
# pos[:, :, 1] = y
print(z([1,2]))

pos=np.dstack((x, y))
z1=z(pos)
print(z1)
ax =fig.add_subplot(projection='3d')
ax.plot_surface(x,y, z(pos), cmap='viridis', linewidth=0)
plt.show()
