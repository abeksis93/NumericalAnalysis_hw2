import sys
from io import StringIO

import numpy as np
from matplotlib import pyplot as plt
from mealpy.math_based.HC import OriginalHC
from mealpy.swarm_based.GWO import BaseGWO


def q1_d1(F, ub, lb):
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
    arrError = []
    print("error:")
    for i, j in zip(list_loss1, arrZ):
        print(min(arrZ) - i)
        arrError.append(-1 * (min(arrZ) - i))
    plt.scatter(arrError, arrIteration, c='red')
    plt.xlabel("X is error value")
    plt.ylabel("Y is number of iteration")
    plt.title("1d - Hill Climber 10D")
    plt.show()


def q1_d2(F, ub, lb):
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