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
    return(1/(1+exp(-z)))


class TrueShade:

    # initialize variable
    # grid_size : grid of datapoints ..[50, 50]
    # input_lim : defined area limit, 1D->[x](0:x), 2D->[x1, x2](0:x1, 0:x2) ..[10, 10]
    # mixture_size : number of mixture 
    # logistic coef : given logistic coefficient a, b (z = ax+b)
    def __init__(self, input_size=2000, input_lim, )
