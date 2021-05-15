import Question1B
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

import Question1C
import Question1D
import Question2B


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


def create_function(n, m, ul, r, w, d):
    w_list = []
    for i in range(0, 6):
        w_list.append(random.random() * r)

    variance_list = []
    miu_list = []

    for i in range(0, m + 1):
        miu_list.append([np.random.uniform(-d*ul/2, d*ul/2) for i in range(n)])
        variance_list.append([np.random.uniform(w*ul/10, w*ul) for i in range(n)])

    return G(m, variance_list, miu_list)


z = create_function(n, m, ul, r, w, d)
variables = []
for i in range(0, n):
    variables.append(np.linspace(-ul / 2, ul / 2, 1000))

# plot
fig = plt.figure()

x, y = np.meshgrid(variables[0], variables[1])

pos = np.dstack((x, y))
zMax = z(pos)
print(np.amax(zMax))

ax = fig.add_subplot(projection='3d')
ax.plot_surface(x, y, z(pos), cmap='viridis', linewidth=0)
plt.show()


# ____define objective function___
F = lambda x: -z(x)

lb = [-ul / 2] * n
ub = [ul / 2] * n

Question1B.q1_b(F, ub, lb)
Question1C.q1_c(F, lb, ub)

# parameter choose:
n = nList[1]

z = create_function(n, m, ul, r, w, d)
variables = []
for i in range(0, n):
    variables.append(np.linspace(-ul / 2, ul / 2, 1000))

lb = [-10.02] * n
ub = [10.02] * n
Question1D.q1_d1(F, ub, lb)

lb = [-7] * n
ub = [6.0001] * n
Question1D.q1_d2(F, ub, lb)


# ___2.b___

# parameter choose:
n = nList[0]  # n == 5
r = nList[1]  # r == 0.3

z = create_function(n, m, ul, r, w, d)
variables = []
for i in range(0, n):
    variables.append(np.linspace(-ul / 2, ul / 2, 1000))

F = lambda x: -z(x)
lb = [-ul / 2] * n
ub = [ul / 2] * n

epoch_num = 50  # Number of iterations
pop_size = 50  # population size
dim_n = 5

Question2B.q2_b(F, ub, lb, epoch_num, dim_n, r)

# parameter choose:
n = nList[0]  # n == 5
r = nList[3]  # r == 0.9

z = create_function(n, m, ul, r, w, d)
variables = []
for i in range(0, n):
    variables.append(np.linspace(-ul / 2, ul / 2, 1000))

F = lambda x: -z(x)
lb = [-ul / 2] * n
ub = [ul / 2] * n
# lb = [-10.02] * n
# ub = [10.02] * n

epoch_num = 50  # Number of iterations
pop_size = 50  # population size
dim_n = 5

Question2B.q2_b(F, ub, lb, epoch_num, dim_n, r)

# parameter choose:
n = nList[1]  # n == 10
r = rList[1]  # r == 0.3

z = create_function(n, m, ul, r, w, d)
variables = []
for i in range(0, n):
    variables.append(np.linspace(-ul / 2, ul / 2, 1000))

F = lambda x: -z(x)
lb = [-ul / 2] * n
ub = [ul / 2] * n

epoch_num = 50  # Number of iterations
pop_size = 50  # population size
dim_n = 10

Question2B.q2_b(F, ub, lb, epoch_num, dim_n, r)

# parameter choose:
n = nList[1]  # n == 10
r = nList[3]  # r == 0.9

z = create_function(n, m, ul, r, w, d)
variables = []
for i in range(0, n):
    variables.append(np.linspace(-ul / 2, ul / 2, 1000))

F = lambda x: -z(x)
lb = [-ul / 2] * n
ub = [ul / 2] * n
# lb = [-10.02] * n
# ub = [10.02] * n

epoch_num = 50  # Number of iterations
pop_size = 50  # population size
dim_n = 10

Question2B.q2_b(F, ub, lb, epoch_num, dim_n, r)
