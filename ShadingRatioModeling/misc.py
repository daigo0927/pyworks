import sys, os
sys.path.append(os.pardir)
import numpy as np
import itertools
from common.functions import sigmoid
from common.gradient import numerical_gradient
from PIL import Image
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d.axes3d import Axes3D


def dnorm(x, mu, sigma):
    return((1/np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2)) )

def dnorm_Mixture(x, mu, sigma, pi):
    values = dnorm(x, mu, sigma)
    return(sum(pi*values))

def sigmoid(z):
    return(1/(1+np.exp(-z)))

def GenerateGMM(mus, covs_fake):
    dim = mus.shape[1]
    norms = {}
    for i in range(mus.shape[0]):
        mu = mus[i, :]
        cov_fake = covs_fake[i, :].reshape(dim, dim)
        cov = np.dot(cov_fake.T, cov_fake)
        norms[str(i)] = multivariate_normal(mean = mu, cov = cov)
    return(norms)

def MixtureValue(x, GMMmodel, pi):
    values = []
    for i in range(len(pi)):
        values.append(GMMmodel[str(i)].pdf(x))
    return(sum(values*pi))

def BregmanDivergence(f, g, a):
    lg1 = np.log((1.-f)/(1.-g))
    lg2 = np.log(g*(1.-f)/f/(1.-g))

    allloss = lg1/a - f * lg2/a
    return(np.mean(allloss))

class Epanechnikov2D:

    # 2 variate Epanechnikov function
    # f(x) = max(0, D-(x-mean)*cov*(x-mean.T))
    # D : normalize constant
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

        h = 1e-4
        a = cov[0,0]
        b = cov[0,1]
        c = cov[1,1]
        if(a == c):
            c = c+h

        theta = 1/2*np.arctan(2*b/(a-c))
        tmp1 = \
            c*np.cos(theta)**2 - 2*b*np.sin(theta)*np.cos(theta) + a*np.sin(theta)**2
        tmp2 = \
            c*np.sin(theta)**2 + 2*b*np.sin(theta)*np.cos(theta) + a*np.cos(theta)**2
        det = a*c-b**2
        self.D = np.sqrt(2 * np.sqrt(tmp1*tmp2)/np.pi/det)

    def pdf(self, x):
        mean = self.mean
        cov = self.cov
        D = self.D

        tmp_value = D - np.dot((x-mean), np.linalg.solve(cov, (x-mean).T))
        density_value = np.diag(tmp_value)
        density_value.flags.writeable = True
        density_value[density_value < 0] = 0
        return(density_value)

class Epanechnikov2Dmix:
    def __init__(self, mus, covs, pi):
        self.mus = mus
        self.covs = covs
        self.pi = pi
        self.mix = pi.shape[0]

        self.Epas = \
        [Epanechnikov2D(mean = mu, cov = cov) for mu, cov in zip(self.mus, self.covs)]

    def pdf(self, x):
        q = np.array([Epa.pdf(x) for Epa in self.Epas])
        q_weighted = q*self.pi.reshape(self.mix, 1)
        q_sum = np.sum(q_weighted, axis = 0)

        return(q_sum)

        
        
    
