import math
import random

# import gaussian as gaussian
import matplotlib.pyplot as plt
import numpy
import numpy as np
from mealpy.math_based.HC import BaseHC, OriginalHC
from mealpy.swarm_based.GWO import BaseGWO, RW_GWO
from scipy.stats import multivariate_normal


def G(m, variance_list, miu_list):
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
        if max_val < g_lst[i].pdf(miu_list[i]):
            max_val = g_lst[i].pdf(miu_list[i])
            f = g_lst[i]


    return lambda x: sum(
        r * (1 - 0.05 * (i+1)) * g_lst[i].pdf(x) if g_lst[i] != f else f.pdf(x) / max_val for i in range(m + 1))


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
    miu_list.append([np.random.uniform(-d*ul/2,d*ul/2) for i in range(n)])
    variance_list.append([np.random.uniform(w*ul/10,w*ul) for i in range(n)])

variables = []
for i in range(0, n):
    variables.append(np.linspace(-ul / 2, ul / 2, 1000))

# function
z = G(m, variance_list, miu_list)

# plot
fig = plt.figure()

x, y = np.meshgrid(variables[0], variables[1])


pos=np.dstack((x, y))
z1=z(pos)
# print(np.amax(z1))
#
ax = fig.add_subplot(projection='3d')
ax.plot_surface(x,y, z1, cmap='viridis', linewidth=0)
# plt.show()

F = lambda x:-z(x)


# ____define objective function___
lb = [-ul / 2] * n
ub = [ul / 2] * n
# _____1.b______
np.random.seed(12)
hc = OriginalHC(F, ub=ub, lb=lb, epoch=10, problem_size=n)
best_pos1, best_fit1, list_loss1 = hc.train()
print("best_pos: ")
print(best_pos1)
print("best_fit: ")
print(best_fit1)
print("list_loss: ")
print(list_loss1)
for i in list_loss1:
    print(i)

# ____1.c____
# 1.c

print("1c")
gwo = RW_GWO(obj_func=F, lb=lb, ub=ub, epoch=10, problem_size=n)
best_pos1, best_fit1, list_loss1 = gwo.train()
print("best_pos: ")
print(best_pos1)
print("best_fit: ")
print(best_fit1)
print("list_loss: ")
for i in list_loss1:
    print(i)

# ___1.d___
print("1d")

n=10
miu_list=[]
variance_list=[]
for i in range(m+1):
    var = [np.random.uniform(w*ul/10,w*ul) for i in range(n)]
    miu=[np.random.uniform(-d*ul/2,d*ul/2) for i in range(n)]
    miu_list.append(miu)
    variance_list.append(var)


z = G(m, variance_list, miu_list)
F = lambda x:-z(x)
lb = [-ul / 2] * n
ub = [ul / 2] * n
# _____hc______
np.random.seed(12)
hc = OriginalHC(F, ub=ub, lb=lb, epoch=50, problem_size=n)
best_pos1, best_fit1, list_loss1 = hc.train()
print("best_pos: ")
print(best_pos1)
print("best_fit: ")
print(best_fit1)
print("list_loss: ")
print(list_loss1)
for i in list_loss1:
    print(i)

# ____gwo____
gwo = RW_GWO(obj_func=F, lb=lb, ub=ub, epoch=50, problem_size=n)
best_pos1, best_fit1, list_loss1 = gwo.train()
print("best_pos: ")
print(best_pos1)
print("best_fit: ")
print(best_fit1)
print("list_loss: ")
for i in list_loss1:
    print(i)