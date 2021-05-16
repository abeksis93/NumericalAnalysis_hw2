import statistics
import matplotlib.pyplot as plt

arr_stdev_HC = [0.004338988950433336, 0.06090100407855699, 6.688962011496488e-10, 0.004338988950433336]
arr_stdev_SA = [0.4872664861331653, 0.22676411708917707, 0.36908616863400245, 0.4872664861331653]
arr_stdev_PSO = [0.018086004984118945, 0.23525587628069922, 5.1261715699186875e-05, 0.018086004984118945]
arr_stdev_WOA = [0.017557026967603234, 0.0007394714621979141, 0.2717192495902559, 0.017557026967603234]
arr_stdev_HS = [0.30598309690333, 0.05209433858086878, 0.00025006849066719664, 0.30598309690333]
arr_stdev_GWO = [0.012304950503158513, 0.00030437217665938397, 0.2904809712818848, 0.012304950503158513]
arr_stdev_EHO = [0.26786573853247736, 0.13054487936939127, 0.00016168399503930147, 0.26786573853247736]

mean_stdev_HC = statistics.mean(arr_stdev_HC)
mean_stdev_SA = statistics.mean(arr_stdev_SA)
mean_stdev_PSO = statistics.mean(arr_stdev_PSO)
mean_stdev_WOA = statistics.mean(arr_stdev_WOA)
mean_stdev_HS = statistics.mean(arr_stdev_HS)
mean_stdev_GWO = statistics.mean(arr_stdev_GWO)
mean_stdev_EHO = statistics.mean(arr_stdev_EHO)


fig = plt.figure()
algo = ['HC', 'SA', 'PSO', 'WOA', 'HS', 'GWO', 'EHO']
mean_stdev_val = []
mean_stdev_val.append(mean_stdev_HC)
mean_stdev_val.append(mean_stdev_SA)
mean_stdev_val.append(mean_stdev_PSO)
mean_stdev_val.append(mean_stdev_WOA)
mean_stdev_val.append(mean_stdev_GWO)
mean_stdev_val.append(mean_stdev_EHO)
mean_stdev_val.append(mean_stdev_HS)
plt.bar(algo, mean_stdev_val, color='maroon', width=0.3)
plt.title("Mean Standard Deviation of algorithms")
plt.show()





