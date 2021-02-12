import numpy as np
from numpy.random import multivariate_normal, uniform
import scipy.stats as stats

# This is a Monte-Carlo Markov Chain (MCMC) class that provides methods for performing
# Metropolis Random walk MCMC and Adaptive Covariance MC.


class MCMC():
    """
    :param jumpFcn: is the function that proposes the next move based on the current
           position.
    :param jumpCov: Covariance for the normally distributed covariance function
    :param posteriorFcn: is the function that is being sampled from. 
    :param posteriorMean: is the mean of the posterior distribution
    :param posteriorCov: is the covariance matrix of the posterior distribution
    """

    # Constructor
    def __init__(self, jumpFcn, jumpCov, posteriorFcn, posteriorMean, posteriorCov):
        self.jumpFcn = jumpFcn
        self.jCov = jumpCov
        self.posteriorFcn = posteriorFcn
        self.pMean = posteriorMean
        self.pCov = posteriorCov

        self.theta_now = None
        self.theta_proposed = None

    def _accrej(self):
        """ 
        Method that accepts or rejects the proposed move
        :return: the current position in the parameter space
        """
        u = uniform()
        r = self.posteriorFcn(self.theta_proposed, self.pMean, self.pCov) / \
            self.posteriorFcn(self.theta_now, self.pMean, self.pCov)
        if(r >= u):
            self.theta_now = self.theta_proposed

        return self.theta_now

    def metropolisRandomWalk(self, X_0, n_samples):
        """ 
        Method that runs the metropolis random walk algorithm
        :param X_0: Array containing the starting points for the chains
        :param n_samples: Number of samples from the target function
        :return: A numpy array containing the path traversed by each chain 
        """
        chains = []
        # Start a chain from each starting position given
        for x0 in X_0:
            self.theta_now = x0
            theta = [self.theta_now]
            # Keep sampling until n_samples obtained
            while(len(theta) != n_samples):
                self.theta_proposed = self.jumpFcn(self.theta_now, self.jCov)
                theta.append(self._accrej())

            chains.append(theta)
        return chains

    def adaptiveCovariance(self, X_0, n_samples=100, startAdapt=0.2, eta=0.5):
        """ 
        Method that runs the metropolis random walk algorithm
        :param X_0: Array containing the starting points for the chains
        :param n_samples: Number of samples from the target function
        :param startAdapt: Fraction of n_samples at which the covariance
               adaptation begins
        :param eta: Exponent for the covariance adaptation
        :return: A numpy array containing the path traversed by each chain 
        """
        adaptIter = np.floor(startAdapt*n_samples)
        chains = []
        # Start a chain from each starting position given
        for x0 in X_0:
            adaptCount = 0
            self.theta_now = x0
            theta = [self.theta_now]
            # Keep sampling until n_samples obtained
            while (len(theta) != n_samples):
                self.theta_proposed = self.jumpFcn(self.theta_now, self.jCov)
                theta.append(self._accrej())
                # Begin updating variance after a set number of iterations
                if adaptCount > adaptIter:
                    gamma = (adaptCount - adaptIter)**-eta
                    self.jCov = gamma*np.cov(theta, rowvar=False)

                adaptCount += 1
            chains.append(theta)
        return chains
