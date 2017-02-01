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

def GenerateGMM(mus, covs):
    dim = mus.shape[1]
    norms = {}
    for i in range(mus.shape[0]):
        mu = mus[i, :]
        cov = covs[i, :].reshape(dim, dim)
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
    
    
    
