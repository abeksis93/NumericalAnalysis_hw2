import sys
from io import StringIO
import numpy as np
from matplotlib import pyplot as plt
from mealpy.swarm_based.GWO import BaseGWO


def q1_c(func, lb, ub):
    np.random.seed(12)
    old_stdout = sys.stdout
    sys.stdout = my_stdout = StringIO()

    gwo = BaseGWO(obj_func=func, lb=lb, ub=ub, epoch=10, problem_size=2)
    best_pos1, best_fit1, list_loss1 = gwo.train()

    sys.stdout = old_stdout
    result_string = my_stdout.getvalue()
    arr_result_string = result_string.split("\n")
    arr_z_string = [arr_result_string[i].split(",") for i in range(10)]
    arr_zin1 = [arr_z_string[i][1].split(":") for i in range(10)]
    arr_z = []
    arr_iteration = []
    for i in range(10):
        arr_z.append(float(arr_zin1[i][1]))
        arr_iteration.append(i)
    all_positive_lst = [abs(num) for num in arr_z]
    plt.scatter(all_positive_lst, arr_iteration, c='green')
    plt.xlabel("X is best fit value")
    plt.ylabel("Y is number of iteration")
    plt.title("1c - Grey Wolf Optimizer 2D")
    plt.show()

