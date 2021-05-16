import sys
from io import StringIO
from random import random

import numpy as np
from matplotlib import pyplot as plt
from mealpy.math_based.HC import OriginalHC
from scipy.stats import multivariate_normal


def q1_b(F, ub, lb):

    np.random.seed(12)
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    hc = OriginalHC(F, ub=ub, lb=lb, verbose=True, epoch=10, problem_size=2)
    best_pos1, best_fit1, list_loss1 = hc.train()

    sys.stdout = old_stdout
    result_string = mystdout.getvalue()
    arr_result_string = result_string.split("\n")
    arr_z_string = [arr_result_string[i].split(",") for i in range(10)]
    arr_zin1 = [arr_z_string[i][1].split(":") for i in range(10)]
    arr_z = []
    arr_iteration = []
    for i in range(10):
        arr_z.append(float(arr_zin1[i][1]))
        arr_iteration.append(i)
    all_positive_lst = [abs(num) for num in arr_z]
    plt.scatter(all_positive_lst, arr_iteration, c='purple')
    plt.xlabel("X is best fit value")
    plt.ylabel("Y is number of iteration")
    plt.title("1b - Hill Climber 2D")
    plt.show()
