import math
import random

# import gaussian as gaussian
import sys
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
from mealpy.math_based.HC import BaseHC, OriginalHC
from mealpy.swarm_based.GWO import BaseGWO
from mealpy.swarm_based.PSO import BasePSO
from mealpy.swarm_based.EHO import BaseEHO
from mealpy.swarm_based.WOA import BaseWOA
from mealpy.physics_based.SA import BaseSA

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
        if max_val < multivariate_normal(miu_list[i], np.diag(variance_list[i])).pdf(miu_list[i]):
            max_val = multivariate_normal(miu_list[i], np.diag(variance_list[i])).pdf(miu_list[i])
    g_list_mu = zip(g_lst, miu_list)
    g_list_mu = sorted(g_list_mu, key=lambda g_mu: g_mu[0].pdf(g_mu[1]), reverse=True)
    g_lst = [g_mu[0] for g_mu in g_list_mu]

    # for j in range(len(g_lst)):
    #     if j != 0:
    #         print((r * (1 - 0.05 * (i - 1))) * g_lst[j].pdf(miu_list[j]))
    #     else:
    #         print(g_lst[0].pdf(miu_list[0]) / max_val)
    return lambda x: sum((r * (1 - 0.05 * (i - 1))) * g_lst[i].pdf(x) if i != 0 else g_lst[0].pdf(x) / max_val for i in range(m + 1))


nList = [5, 10, 20, 40]
mList = [0, 5, 10, 20]
rList = [0.1, 0.3, 0.6, 0.9]
ulList = [10, 20, 30, 40]
wList = [0.01, 0.03, 0.06, 0.09]
dList = [0.25, 0.5, 0.75, 1.0]

# parameter choose:
n = 2
m = mList[1]
ul = ulList[1]
r = rList[1]
w = wList[1]
d = dList[1]
Wlist = []
for i in range(0, 6):
    Wlist.append(random.random() * r)


variance_list = []
miu_list = []

for i in range(0, m + 1):
    miu_list.append([np.random.uniform(-d*ul/2, d*ul/2) for i in range(n)])
    variance_list.append([np.random.uniform(w*ul/10, w*ul) for i in range(n)])

variables = []
for i in range(0, n):
    variables.append(np.linspace(-ul / 2, ul / 2, 1000))


# function
z = G(m, variance_list, miu_list)

# plot
fig = plt.figure()

x, y = np.meshgrid(variables[0], variables[1])

pos = np.dstack((x, y))
zMax = z(pos)
print(np.amax(zMax))


ax = fig.add_subplot(projection='3d')
ax.plot_surface(x, y, z(pos), cmap='viridis', linewidth=0)
plt.show()

F = lambda x: -z(x)

# ____define objective function___
lb = [-ul / 2] * n
ub = [ul / 2] * n

# _____1.b______
np.random.seed(12)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

hc = OriginalHC(F, ub=ub, lb=lb, verbose=True, epoch=10, problem_size=2)
best_pos1, best_fit1, list_loss1 = hc.train()

sys.stdout = old_stdout
ResultString = mystdout.getvalue()
arrResultString = ResultString.split("\n")
arrZString = [arrResultString[i].split(",") for i in range(10)]
arrZin1 = [arrZString[i][1].split(":") for i in range(10)]
arrZ = []
arrIteration = []
for i in range(10):
    arrZ.append(float(arrZin1[i][1]))
    arrIteration.append(i)
plt.scatter(arrZ, arrIteration, c='purple')
plt.xlabel("X is best fit value")
plt.ylabel("Y is number of iteration")
plt.title("1b - Hill Climber 2D")
plt.show()

# ____1.c____
print("1c")
np.random.seed(12)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

gwo = BaseGWO(obj_func=F, lb=lb, ub=ub, epoch=10, problem_size=2)
best_pos1, best_fit1, list_loss1 = gwo.train()

sys.stdout = old_stdout
ResultString = mystdout.getvalue()
arrResultString = ResultString.split("\n")
arrZString = [arrResultString[i].split(",") for i in range(10)]
arrZin1 = [arrZString[i][1].split(":") for i in range(10)]
arrZ = []
arrIteration = []
for i in range(10):
    arrZ.append(float(arrZin1[i][1]))
    arrIteration.append(i)
plt.scatter(arrZ, arrIteration, c='green')
plt.xlabel("X is best fit value")
plt.ylabel("Y is number of iteration")
plt.title("1c - Grey Wolf Optimizer 2D")
plt.show()


# ___1.d___
print("1d")
nList = [5, 10, 20, 40]
mList = [0, 5, 10, 20]
rList = [0.1, 0.3, 0.6, 0.9]
ulList = [10, 20, 30, 40]
wList = [0.01, 0.03, 0.06, 0.09]
dList = [0.25, 0.5, 0.75, 1.0]

# parameter choose:
n = nList[1]
m = mList[1]
ul = ulList[1]
r = rList[1]
w = wList[1]
d = dList[1]
Wlist = []
for i in range(0, 6):
    Wlist.append(random.random() * r)

miu_list = []
variance_list = []

for i in range(0, m + 1):
    miu_list.append([np.random.uniform(-d*ul/2, d*ul/2) for i in range(n)])
    variance_list.append([np.random.uniform(w*ul/10, w*ul) for i in range(n)])

z = G(m, variance_list, miu_list)
F = lambda x: -z(x)
lb = [-10.02] * n
ub = [10.02] * n
# _____hc______
np.random.seed(12)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

hc = OriginalHC(F, ub=ub, lb=lb, epoch=50, problem_size=10)
best_pos1, best_fit1, list_loss1 = hc.train()

sys.stdout = old_stdout
ResultString = mystdout.getvalue()
arrResultString = ResultString.split("\n")
arrZString = [arrResultString[i].split(",") for i in range(50)]
arrZin1 = [arrZString[i][1].split(":") for i in range(50)]
arrZ = []
arrIteration = []
for i in range(50):
    arrZ.append(float(arrZin1[i][1]))
    arrIteration.append(i)
# plt.scatter(arrZ, arrIteration)
# plt.show()

arrError = []
print("error:")
for i, j in zip(list_loss1, arrZ):
    print(min(arrZ) - i)
    arrError.append(min(arrZ) - i)
plt.scatter(arrError, arrIteration, c='red')
plt.xlabel("X is error value")
plt.ylabel("Y is number of iteration")
plt.title("1d - Hill Climber 10D")
plt.show()


lb = [-7] * n
ub = [6.0001] * n
# ____gwo____
np.random.seed(12)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

gwo = BaseGWO(obj_func=F, lb=lb, ub=ub, epoch=50, problem_size=10)
best_pos1, best_fit1, list_loss1 = gwo.train()

sys.stdout = old_stdout
ResultString = mystdout.getvalue()
arrResultString = ResultString.split("\n")
arrZString = [arrResultString[i].split(",") for i in range(50)]
arrZin1 = [arrZString[i][1].split(":") for i in range(50)]
arrZ = []
arrIteration = []
for i in range(50):
    arrZ.append(float(arrZin1[i][1]))
    print(arrZ[i])
    arrIteration.append(i)

arrError = []
print("error:")
for i, j in zip(list_loss1, arrZ):
    print(min(arrZ) - i)
    arrError.append(min(arrZ) - i)
plt.scatter(arrError, arrIteration, c='orange')
plt.xlabel("X is error value")
plt.ylabel("Y is number of iteration")
plt.title("1d - Grey Wolf Optimizer 10D")
plt.show()


# ___2.b___
print("2b")
nList = [5, 10]
mList = [0, 5]
rList = [0.3, 0.9]
ulList = [10, 20]
wList = [0.01, 0.03]
dList = [0.25, 0.5]

# parameter choose:
n = nList[0]
m = mList[0]
ul = ulList[0]
r = rList[0]
w = wList[0]
d = dList[0]

Wlist = []
for i in range(0, 6):
    Wlist.append(random.random() * r)

miu_list = []
variance_list = []

for i in range(0, m + 1):
    miu_list.append([np.random.uniform(-d*ul/2, d*ul/2) for i in range(n)])
    variance_list.append([np.random.uniform(w*ul/10, w*ul) for i in range(n)])

z = G(m, variance_list, miu_list)
F = lambda x: -z(x)
lb = [-10.02] * n
ub = [10.02] * n

# ______ Number of iterations _______
epoch_num = 10

# ______ 5D _______
dim_n = 5

# _____hc 5D______
np.random.seed(12)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

hc = OriginalHC(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
best_pos1, best_fit1, list_loss1 = hc.train()

sys.stdout = old_stdout
ResultString = mystdout.getvalue()
arrResultString = ResultString.split("\n")
arrZString = [arrResultString[i].split(",") for i in range(epoch_num)]
arrZin1 = [arrZString[i][1].split(":") for i in range(epoch_num)]
arrZ = []
arrIteration = []
for i in range(epoch_num):
    arrZ.append(float(arrZin1[i][1]))
    arrIteration.append(i)
# plt.scatter(arrZ, arrIteration)
# plt.show()

arrError = []
print("error:")
for i, j in zip(list_loss1, arrZ):
    print(min(arrZ) - i)
    arrError.append(min(arrZ) - i)
plt.scatter(arrError, arrIteration, c='red')
plt.xlabel("X is error value")
plt.ylabel("Y is number of iteration")
plt.title("2b - Hill Climber 5D")
plt.show()


# _____SA 5D______
np.random.seed(12)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

hc = BaseSA(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
best_pos1, best_fit1, list_loss1 = hc.train()

sys.stdout = old_stdout
ResultString = mystdout.getvalue()
arrResultString = ResultString.split("\n")
arrZString = [arrResultString[i].split(",") for i in range(epoch_num)]
arrZin1 = [arrZString[i][1].split(":") for i in range(epoch_num)]
arrZ = []
arrIteration = []
for i in range(epoch_num):
    arrZ.append(float(arrZin1[i][1]))
    arrIteration.append(i)
# plt.scatter(arrZ, arrIteration)
# plt.show()

arrError = []
print("error:")
for i, j in zip(list_loss1, arrZ):
    print(min(arrZ) - i)
    arrError.append(min(arrZ) - i)
plt.scatter(arrError, arrIteration, c='red')
plt.xlabel("X is error value")
plt.ylabel("Y is number of iteration")
plt.title("2b - Simulated Annealing 5D")
plt.show()


# _____PSO 5D______
np.random.seed(12)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

hc = BasePSO(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
best_pos1, best_fit1, list_loss1 = hc.train()

sys.stdout = old_stdout
ResultString = mystdout.getvalue()
arrResultString = ResultString.split("\n")
arrZString = [arrResultString[i].split(",") for i in range(epoch_num)]
arrZin1 = [arrZString[i][1].split(":") for i in range(epoch_num)]
arrZ = []
arrIteration = []
for i in range(epoch_num):
    arrZ.append(float(arrZin1[i][1]))
    arrIteration.append(i)
# plt.scatter(arrZ, arrIteration)
# plt.show()

arrError = []
print("error:")
for i, j in zip(list_loss1, arrZ):
    print(min(arrZ) - i)
    arrError.append(min(arrZ) - i)
plt.scatter(arrError, arrIteration, c='red')
plt.xlabel("X is error value")
plt.ylabel("Y is number of iteration")
plt.title("2b - Particle Swarm Optimization 5D")
plt.show()

# _____EHO 5D______
np.random.seed(12)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

hc = BaseEHO(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
best_pos1, best_fit1, list_loss1 = hc.train()

sys.stdout = old_stdout
ResultString = mystdout.getvalue()
arrResultString = ResultString.split("\n")
arrZString = [arrResultString[i].split(",") for i in range(epoch_num)]
arrZin1 = [arrZString[i][1].split(":") for i in range(epoch_num)]
arrZ = []
arrIteration = []
for i in range(epoch_num):
    arrZ.append(float(arrZin1[i][1]))
    arrIteration.append(i)
# plt.scatter(arrZ, arrIteration)
# plt.show()

arrError = []
print("error:")
for i, j in zip(list_loss1, arrZ):
    print(min(arrZ) - i)
    arrError.append(min(arrZ) - i)
plt.scatter(arrError, arrIteration, c='red')
plt.xlabel("X is error value")
plt.ylabel("Y is number of iteration")
plt.title("2b - Elephant Herding Optimization 5D")
plt.show()

# _____WOA 5D______
np.random.seed(12)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

hc = BaseWOA(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
best_pos1, best_fit1, list_loss1 = hc.train()

sys.stdout = old_stdout
ResultString = mystdout.getvalue()
arrResultString = ResultString.split("\n")
arrZString = [arrResultString[i].split(",") for i in range(epoch_num)]
arrZin1 = [arrZString[i][1].split(":") for i in range(epoch_num)]
arrZ = []
arrIteration = []
for i in range(epoch_num):
    arrZ.append(float(arrZin1[i][1]))
    arrIteration.append(i)
# plt.scatter(arrZ, arrIteration)
# plt.show()

arrError = []
print("error:")
for i, j in zip(list_loss1, arrZ):
    print(min(arrZ) - i)
    arrError.append(min(arrZ) - i)
plt.scatter(arrError, arrIteration, c='red')
plt.xlabel("X is error value")
plt.ylabel("Y is number of iteration")
plt.title("2b - Whale Optimization Algorithm 5D")
plt.show()

# _____GWO 5D______
np.random.seed(12)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

hc = BaseGWO(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
best_pos1, best_fit1, list_loss1 = hc.train()

sys.stdout = old_stdout
ResultString = mystdout.getvalue()
arrResultString = ResultString.split("\n")
arrZString = [arrResultString[i].split(",") for i in range(epoch_num)]
arrZin1 = [arrZString[i][1].split(":") for i in range(epoch_num)]
arrZ = []
arrIteration = []
for i in range(epoch_num):
    arrZ.append(float(arrZin1[i][1]))
    arrIteration.append(i)
# plt.scatter(arrZ, arrIteration)
# plt.show()

arrError = []
print("error:")
for i, j in zip(list_loss1, arrZ):
    print(min(arrZ) - i)
    arrError.append(min(arrZ) - i)
plt.scatter(arrError, arrIteration, c='red')
plt.xlabel("X is error value")
plt.ylabel("Y is number of iteration")
plt.title("2b - Grey Wolf Optimizer 5D")
plt.show()


# parameter choose:
n = nList[1]
m = mList[1]
ul = ulList[1]
r = rList[1]
w = wList[1]
d = dList[1]
Wlist = []
for i in range(0, 6):
    Wlist.append(random.random() * r)

miu_list = []
variance_list = []

for i in range(0, m + 1):
    miu_list.append([np.random.uniform(-d*ul/2, d*ul/2) for i in range(n)])
    variance_list.append([np.random.uniform(w*ul/10, w*ul) for i in range(n)])

z = G(m, variance_list, miu_list)
F = lambda x: -z(x)
lb = [-10.02] * n
ub = [10.02] * n

# ______ 10D _______
dim_n = 10

# _____hc 10D______
np.random.seed(12)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

hc = OriginalHC(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
best_pos1, best_fit1, list_loss1 = hc.train()

sys.stdout = old_stdout
ResultString = mystdout.getvalue()
arrResultString = ResultString.split("\n")
arrZString = [arrResultString[i].split(",") for i in range(epoch_num)]
arrZin1 = [arrZString[i][1].split(":") for i in range(epoch_num)]
arrZ = []
arrIteration = []
for i in range(epoch_num):
    arrZ.append(float(arrZin1[i][1]))
    arrIteration.append(i)
# plt.scatter(arrZ, arrIteration)
# plt.show()

arrError = []
print("error:")
for i, j in zip(list_loss1, arrZ):
    print(min(arrZ) - i)
    arrError.append(min(arrZ) - i)
plt.scatter(arrError, arrIteration, c='red')
plt.xlabel("X is error value")
plt.ylabel("Y is number of iteration")
plt.title("2b - Hill Climber 10D")
plt.show()

# _____SA 10D______
np.random.seed(12)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

hc = BaseSA(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
best_pos1, best_fit1, list_loss1 = hc.train()

sys.stdout = old_stdout
ResultString = mystdout.getvalue()
arrResultString = ResultString.split("\n")
arrZString = [arrResultString[i].split(",") for i in range(epoch_num)]
arrZin1 = [arrZString[i][1].split(":") for i in range(epoch_num)]
arrZ = []
arrIteration = []
for i in range(epoch_num):
    arrZ.append(float(arrZin1[i][1]))
    arrIteration.append(i)
# plt.scatter(arrZ, arrIteration)
# plt.show()

arrError = []
print("error:")
for i, j in zip(list_loss1, arrZ):
    print(min(arrZ) - i)
    arrError.append(min(arrZ) - i)
plt.scatter(arrError, arrIteration, c='red')
plt.xlabel("X is error value")
plt.ylabel("Y is number of iteration")
plt.title("2b - Simulated Annealing 10D")
plt.show()


# _____PSO 10D______
np.random.seed(12)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

hc = BasePSO(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
best_pos1, best_fit1, list_loss1 = hc.train()

sys.stdout = old_stdout
ResultString = mystdout.getvalue()
arrResultString = ResultString.split("\n")
arrZString = [arrResultString[i].split(",") for i in range(epoch_num)]
arrZin1 = [arrZString[i][1].split(":") for i in range(epoch_num)]
arrZ = []
arrIteration = []
for i in range(epoch_num):
    arrZ.append(float(arrZin1[i][1]))
    arrIteration.append(i)
# plt.scatter(arrZ, arrIteration)
# plt.show()

arrError = []
print("error:")
for i, j in zip(list_loss1, arrZ):
    print(min(arrZ) - i)
    arrError.append(min(arrZ) - i)
plt.scatter(arrError, arrIteration, c='red')
plt.xlabel("X is error value")
plt.ylabel("Y is number of iteration")
plt.title("2b - Particle Swarm Optimization 10D")
plt.show()

# _____EHO 10D______
np.random.seed(12)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

hc = BaseEHO(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
best_pos1, best_fit1, list_loss1 = hc.train()

sys.stdout = old_stdout
ResultString = mystdout.getvalue()
arrResultString = ResultString.split("\n")
arrZString = [arrResultString[i].split(",") for i in range(epoch_num)]
arrZin1 = [arrZString[i][1].split(":") for i in range(epoch_num)]
arrZ = []
arrIteration = []
for i in range(epoch_num):
    arrZ.append(float(arrZin1[i][1]))
    arrIteration.append(i)
# plt.scatter(arrZ, arrIteration)
# plt.show()

arrError = []
print("error:")
for i, j in zip(list_loss1, arrZ):
    print(min(arrZ) - i)
    arrError.append(min(arrZ) - i)
plt.scatter(arrError, arrIteration, c='red')
plt.xlabel("X is error value")
plt.ylabel("Y is number of iteration")
plt.title("2b - Elephant Herding Optimization 10D")
plt.show()

# _____WOA 10D______
np.random.seed(12)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

hc = BaseWOA(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
best_pos1, best_fit1, list_loss1 = hc.train()

sys.stdout = old_stdout
ResultString = mystdout.getvalue()
arrResultString = ResultString.split("\n")
arrZString = [arrResultString[i].split(",") for i in range(epoch_num)]
arrZin1 = [arrZString[i][1].split(":") for i in range(epoch_num)]
arrZ = []
arrIteration = []
for i in range(epoch_num):
    arrZ.append(float(arrZin1[i][1]))
    arrIteration.append(i)
# plt.scatter(arrZ, arrIteration)
# plt.show()

arrError = []
print("error:")
for i, j in zip(list_loss1, arrZ):
    print(min(arrZ) - i)
    arrError.append(min(arrZ) - i)
plt.scatter(arrError, arrIteration, c='red')
plt.xlabel("X is error value")
plt.ylabel("Y is number of iteration")
plt.title("2b - Whale Optimization Algorithm 10D")
plt.show()

# _____GWO 10D______
np.random.seed(12)
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

hc = BaseGWO(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
best_pos1, best_fit1, list_loss1 = hc.train()

sys.stdout = old_stdout
ResultString = mystdout.getvalue()
arrResultString = ResultString.split("\n")
arrZString = [arrResultString[i].split(",") for i in range(epoch_num)]
arrZin1 = [arrZString[i][1].split(":") for i in range(epoch_num)]
arrZ = []
arrIteration = []
for i in range(epoch_num):
    arrZ.append(float(arrZin1[i][1]))
    arrIteration.append(i)
# plt.scatter(arrZ, arrIteration)
# plt.show()

arrError = []
print("error:")
for i, j in zip(list_loss1, arrZ):
    print(min(arrZ) - i)
    arrError.append(min(arrZ) - i)
plt.scatter(arrError, arrIteration, c='red')
plt.xlabel("X is error value")
plt.ylabel("Y is number of iteration")
plt.title("2b - Grey Wolf Optimizer 10D")
plt.show()
