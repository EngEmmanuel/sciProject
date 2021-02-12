import numpy as np
import scipy.stats as stats
from MCMC import MCMC
from numpy.random import multivariate_normal


def dataPack(theta):
    # Packs the data ready for animation
    chainX = []
    chainY = []
    for chain in theta:
        x, y = zip(*chain)
        chainX.append(x)
        chainY.append(y)

    chainX = np.array(chainX).T
    chainY = np.array(chainY).T
    data = np.array([chainX, chainY])
    return data


def jumpFcn(x, cov):
    # Samples from a MV normal dist
    return multivariate_normal(x, cov)


def posteriorFcn(x, mean, cov):
    # Calculates y = pdf(x)
    return stats.multivariate_normal.pdf(x, mean, cov)


# Define parameters for the distributions
pMean = np.array([5, 5])
pCov = np.array([[1, 1], [0, 4]])
jCov = np.array([[2, 0], [0, 2]])

# Run the algorithms
n_samples = 1000
X_0 = np.array([[10, 5], [4, 5], [8, 0]])

rwm = MCMC(jumpFcn, jCov, posteriorFcn, pMean, pCov)
thetaRwm = rwm.metropolisRandomWalk(X_0, n_samples)

adCov = MCMC(jumpFcn, jCov, posteriorFcn, pMean, pCov)
thetaCov = adCov.adaptiveCovariance(X_0, n_samples)

thetaRwm = dataPack(thetaRwm)
thetaCov = dataPack(thetaCov)

# Save the data for use in animation.py
np.save("rwm.npy", thetaRwm)
np.save("adapt.npy", thetaCov)
