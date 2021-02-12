import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib import animation
from MCMC import MCMC

# Unpack the chain data
xRwm, yRwm = np.load("rwm.npy")
xCov, yCov = np.load("adapt.npy")

# Define parameters for the distributions
pMean = np.array([5, 5])
pCov = np.array([[1, 1], [0, 4]])

# plot
xrang = [0, 11]
yrang = [-1, 11]

x, y = np.mgrid[xrang[0]:xrang[1]:0.01, yrang[0]:yrang[1]:0.01]
pos = np.dstack((x, y))
rv = stats.multivariate_normal(pMean, pCov)

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)

axs[0].contourf(x, y, rv.pdf(pos))
axs[0].set_xlim(xrang[0], xrang[1])
axs[0].set_ylim(yrang[0], yrang[1])
axs[0].set_title('Target Distribution')
axs[0].set_ylabel('y')

axs[1].contour(x, y, rv.pdf(pos))
axs[1].set_title('Random Walk Metropolis')
axs[1].set_ylabel('y')

axs[2].contour(x, y, rv.pdf(pos))
axs[2].set_title('Adaptive Covariance MCMC')
axs[2].set_ylabel('y')
axs[2].set_xlabel('x')

# Animate
batch_size = 10


def animate(i):
    colors = ['r', 'm', 'b', 'g', 'y', 'k']
    for j in range(xRwm.shape[1]):
        axs[1].scatter(xRwm[:batch_size*i, j],
                       yRwm[:batch_size*i, j], color=colors[j], s=5)
        axs[2].scatter(xCov[:batch_size*i, j],
                       yCov[:batch_size*i, j], color=colors[j], s=5)


ani = animation.FuncAnimation(fig, animate, interval=int(len(xRwm)/batch_size))
plt.show()
