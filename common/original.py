import numpy as np
from scipy.stats import multivariate_normal

def GenerateGMM(mus, covs):    
    dim = mus.shape[1]
    norms = {}
    for i in range(mus.shape[0]):
        mu = mus[i, :]
        cov = covs[i, :].reshape(dim,  dim)
        norms[str(i)] = multivariate_normal(mean = mu, cov=cov)
    return(norms)
