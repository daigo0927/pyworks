import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import itertools
from common.functions import sigmoid
from common.gradient import numerical_gradient
from PIL import Image
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d.axes3d import Axes3D
from misc import *
import pdb

class GMMTrue:
    np.random.seed(1)

    # initialize variable
    # grid_size : grid of datapoints ..[50, 50]
    # input_lim : defined area limit, 1D->[x](0:x), 2D->[x1, x2](0:x1, 0:x2) ..[10, 10]
    # mixture_size : number of mixture 
    # logistic coef : given logistic coefficient a, b (z = ax+b)
    def __init__(self,
                 grid_size = np.array([50, 50]),
                 input_lim = np.array([10, 10]),
                 mixture_size = 3,
                 frame_num = 5,
                 logistic_coefficient = np.array([50., 0])):
        
        self.dimension = len(np.array(input_lim))
        self.mix = mixture_size
        self.grid = grid_size
        
        xgrid = np.linspace(0, input_lim[0], grid_size[0])
        ygrid = np.linspace(0, input_lim[1], grid_size[1])
        xy = []
        for i in xgrid:
            for j in ygrid:
                xy.append([i, j])
        xy = np.array(xy)
        self.xy = xy
        
        self.frame = np.arange(frame_num)

        self.TrueParams = {}
        self.TrueParams['mus'] = np.random.rand(mixture_size, self.dimension)*(input_lim/2)
        tmp = np.identity(self.dimension)
        tmp = np.tile(tmp, (mixture_size, 1))
        cov_fake = tmp*input_lim/3
        self.TrueParams['covs_fake'] = cov_fake.reshape(mixture_size, self.dimension**2)
        self.TrueParams['pi'] = np.random.dirichlet([3]*mixture_size)
        self.TrueParams['move'] = np.random.rand(mixture_size, self.dimension)*(input_lim/frame_num/2)

        self.logistic_coefficient = logistic_coefficient

    def GenerateFrame(self, frame):
        t = frame
        xy = self.xy
        mus = self.TrueParams['mus'].reshape(self.mix, self.dimension)
        covs_fake = self.TrueParams['covs_fake'].reshape(self.mix, self.dimension**2)
        pi = self.TrueParams['pi']
        a, b = self.logistic_coefficient
        
        # cloud moving
        move = self.TrueParams['move'].reshape(self.mix, self.dimension)*t
        mus = mus + move

        mixture = GenerateGMM(mus=mus, covs_fake=covs_fake)
        q = []
        [q.append(MixtureValue(i, GMMmodel=mixture, pi = pi)) for i in xy]
        q = np.array(q) + np.random.normal(0, 0.01)
        g = sigmoid(a*q+b)

        return(np.array(g))

    def Generate(self):
        z = []
        [z.append(self.GenerateFrame(frame = f)) for f in self.frame]
        return(np.array(z))

    def ShadePlot(self, frame_num=0, axtype='wireframe'):
        fig = plt.figure()
        ax = Axes3D(fig)
        x = self.xy[:, 0]
        y = self.xy[:, 1]
        z = self.Generate()[frame_num]
        xgrid = x.reshape(self.grid[0], self.grid[1])
        ygrid = y.reshape(self.grid[0], self.grid[1])
        zgrid = z.reshape(self.grid[0], self.grid[1])

        if(axtype == 'wireframe'): ax.plot_wireframe(x, y, z)
        elif(axtype == 'contour'): ax.contour3D(xgrid, ygrid, zgrid)
        elif(axtype == 'contourf'): ax.contourf3D(xgrid, ygrid, zgrid)
        plt.show()


class EpnTrue:
    np.random.seed(3)
    
    # constructer
    def __init__(self,
                 grid_size = np.array([50, 50]),
                 input_lim = np.array([10, 10]),
                 mixture_size = 3,
                 frame_num = 5,
                 logistic_coefficient = np.array([50, 0])
                 ):

        self.dimension = input_lim.shape[0] # normally 2
        self.mix = mixture_size
        self.grid = grid_size

        xgrid = np.linspace(0, input_lim[0], grid_size[0])
        ygrid = np.linspace(0, input_lim[1], grid_size[1])
        xy = np.empty((0, self.dimension), float)
        for x in xgrid:
            for y in ygrid:
                xy = np.append(xy, np.array([[x, y]]), axis = 0)
                
        self.xy = xy

        self.frame = np.arange(frame_num)

        self.Trueparams = {}
        self.Trueparams['mus'] = \
                    np.random.rand(mixture_size, self.dimension)*(input_lim/2)
        tmps = np.random.rand(self.mix, self.dimension, self.dimension) - 0.5
        tmps = np.identity(self.dimension) + tmps*input_lim/10
        covs = np.array([np.dot(tmp.T, tmp) for tmp in tmps])
        self.Trueparams['covs'] = covs
        self.Trueparams['pi'] = np.random.dirichlet([3]*self.mix)
        self.Trueparams['move'] = \
                    np.random.rand(self.mix, self.dimension)*(input_lim/frame_num/2)

        self.logistic_coefficient = logistic_coefficient

    def GenerateFrame(self, frame):
        f = frame
        xy = self.xy
        mus = self.Trueparams['mus']
        covs = self.Trueparams['covs']
        pi = self.Trueparams['pi']
        a, b = self.logistic_coefficient

        move = self.Trueparams['move'] * f
        mus = mus + move

        Epas = Epanechnikov2Dmix(mus = mus, covs = covs, pi = pi)

        q = Epas.pdf(xy)
        q_noised = q + np.random.normal(0, 0.01)
        g = sigmoid(a * q_noised + b)

        return g

    def Generate(self):
        Frames = np.array([self.GenerateFrame(frame = f) for f in self.frame])
        return Frames
        
    def ShadePlot(self, frame=0, axtype='wireframe'):
        fig = plt.figure()
        ax = Axes3D(fig)
        x = self.xy[:, 0]
        y = self.xy[:, 1]
        z = self.Generate()[frame]
        xgrid = x.reshape(self.grid[0], self.grid[1])
        ygrid = y.reshape(self.grid[0], self.grid[1])
        zgrid = z.reshape(self.grid[0], self.grid[1])

        if(axtype == 'wireframe'): ax.plot_wireframe(x, y, z)
        elif(axtype == 'contour'): ax.contour3D(xgrid, ygrid, zgrid)
        elif(axtype == 'contourf'): ax.contourf3D(xgrid, ygrid, zgrid)
        plt.show()

        
        
        
        



        
        
        
        

        



        
