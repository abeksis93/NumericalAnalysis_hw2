import statistics
import sys
from io import StringIO

import numpy as np
from mealpy.math_based.HC import OriginalHC
from mealpy.music_based.HS import OriginalHS
from mealpy.physics_based.SA import BaseSA
from mealpy.swarm_based.EHO import BaseEHO
from mealpy.swarm_based.GWO import BaseGWO
from mealpy.swarm_based.PSO import BasePSO
from mealpy.swarm_based.WOA import BaseWOA


def mean_and_stdev(arr_mean, arr_stdev, algorithm_name):
    print("algorithm: {0}\n", algorithm_name)
    print("mean: {0}\n".format(statistics.mean(arr_mean)))
    print("stdev: {0}\n".format(statistics.stdev(arr_stdev)))


def q2_b(F, ub, lb, epoch_num, dim_n, r):
    arr_mean_HC = []
    arr_mean_SA = []
    arr_mean_PSO = []
    arr_mean_EHO = []
    arr_mean_WOA = []
    arr_mean_GWO = []
    arr_mean_HS = []
    arr_stdev_HC = []
    arr_stdev_SA = []
    arr_stdev_PSO = []
    arr_stdev_EHO = []
    arr_stdev_WOA = []
    arr_stdev_GWO = []
    arr_stdev_HS = []
    for i in range(10):
        np.random.seed(12)
        print("\nseed {0}\n".format(i))
        print("r equals {0}\n".format(r))

        # _____hc______
        print("__________HC {0}D__________\n".format(dim_n))

        hc = OriginalHC(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
        best_pos1, best_fit1, list_loss1 = hc.train()

        arr_mean_HC.append(best_fit1)
        arr_stdev_HC.append(best_fit1)

        # _____SA______
        print("\n__________SA {0}D__________\n".format(dim_n))

        hc = BaseSA(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
        best_pos1, best_fit1, list_loss1 = hc.train()

        arr_mean_SA.append(best_fit1)
        arr_stdev_SA.append(best_fit1)

        # _____PSO______
        print("\n__________PSO {0}D__________\n".format(dim_n))

        hc = BasePSO(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
        best_pos1, best_fit1, list_loss1 = hc.train()

        arr_mean_PSO.append(best_fit1)
        arr_stdev_PSO.append(best_fit1)

        # _____EHO______
        print("\n__________EHO {0}D__________\n".format(dim_n))

        hc = BaseEHO(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
        best_pos1, best_fit1, list_loss1 = hc.train()

        arr_mean_EHO.append(best_fit1)
        arr_stdev_EHO.append(best_fit1)

        # _____WOA______
        print("\n__________WOA {0}D__________\n".format(dim_n))

        hc = BaseWOA(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
        best_pos1, best_fit1, list_loss1 = hc.train()

        arr_mean_WOA.append(best_fit1)
        arr_stdev_WOA.append(best_fit1)


        # _____GWO______
        print("\n__________GWO {0}D__________\n".format(dim_n))

        hc = BaseGWO(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
        best_pos1, best_fit1, list_loss1 = hc.train()

        arr_mean_GWO.append(best_fit1)
        arr_stdev_GWO.append(best_fit1)


        # _____Harmony Search______
        print("\n__________Harmony Search {0}D__________\n".format(dim_n))

        hs = OriginalHS(F, ub=ub, lb=lb, epoch=epoch_num, problem_size=dim_n)
        best_pos1, best_fit1, list_loss1 = hs.train()

        arr_mean_HS.append(best_fit1)
        arr_stdev_HS.append(best_fit1)

    mean_and_stdev(arr_mean_HC, arr_stdev_HC, "HC")
    mean_and_stdev(arr_mean_SA, arr_stdev_SA, "SA")
    mean_and_stdev(arr_mean_PSO, arr_stdev_PSO, "PSO")
    mean_and_stdev(arr_mean_WOA, arr_stdev_WOA, "WOA")
    mean_and_stdev(arr_mean_HS, arr_stdev_HS, "HS")
    mean_and_stdev(arr_mean_GWO, arr_stdev_GWO, "GWO")
    mean_and_stdev(arr_mean_EHO, arr_stdev_EHO, "EHO")
