# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import multivariate_normal
# mu_x = 0
# variance_x = 3
# #
# mu_y = 0
# variance_y = 15
#
#
# x = np.linspace(-10,10,500)
# y = np.linspace(-10,10,500)
# X,Y = np.meshgrid(x,y)
# print(X.shape+(2,))
# pos = np.empty(X.shape + (2,))
# pos[:, :, 0] = X; pos[:, :, 1] = Y
# rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
#
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')
# plt.show()
# #
import numpy as np

nx, ny = (3, 2)
x = np.linspace(0, 1, 3)
y = np.linspace(0, 2, 3)
xv, yv = np.meshgrid(x, y)
print(yv)