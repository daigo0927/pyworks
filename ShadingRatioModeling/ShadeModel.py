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

class ShadeModel:

    def __init__(self,
                 grid_size = np.array([50, 50]),
                 input_lim = np.array([10, 10]),
                 mixture_size = 10,
                 frame_num = 5,
                 logistic_coefficient = np.array([50, 0])
                 ):
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

        self.params = {}
        self.params['mus'] = np.random.rand(mixture_size, self.dimension)*(input_lim)
        tmp = np.identity(self.dimension)
        tmp = np.tile(tmp, (mixture_size, 1))
        cov_fake = tmp*input_lim/10
        self.params['covs_fake'] = cov_fake.reshape(mixture_size, self.dimension**2)
        weight = np.ones(mixture_size)
        self.params['pi'] = weight/sum(weight)
        self.params['move'] = np.random.rand(mixture_size, self.dimension)*(input_lim/frame_num/5)
        
        self.logistic_coefficient = logistic_coefficient

    def predict(self, x, frame):
        t = frame
        
        mus = self.params['mus'].reshape(self.mix, self.dimension)
        covs_fake = self.params['covs_fake'].reshape(self.mix, self.dimension**2)
        pi = self.params['pi']
        a, b = self.logistic_coefficient
        
        # cloud moving
        move = self.params['move'].reshape(self.mix, self.dimension)*t
        mus = mus + move
        
        mixture = GenerateGMM(mus = mus, covs_fake = covs_fake)
        q = []
        [q.append(MixtureValue(i, GMMmodel=mixture, pi = pi)) for i in x]
        q = np.array(q)
        g = sigmoid(a*q+b)
        
        return(np.array(g))
    
    def loss(self, x, frame, f):
        g = self.predict(x, frame)
        BregmanDiv = BregmanDivergence(f, g, a = self.logistic_coefficient[0])
        return(BregmanDiv)
    
    def Gradient(self, x, frame, f):
        loss_param = lambda param: self.loss(x, frame, f)
        
        grads = {}
        for key in self.params.keys():
            grads[key] = numerical_gradient(loss_param, self.params[key])
            
        return(grads)
        
    def ModelPlot(self, frame=0, axtype='wireframe'):
        fig = plt.figure()
        ax = Axes3D(fig)
        x = self.xy[:, 0]
        y = self.xy[:, 1]
        z = self.predict(self.xy, frame=frame)
        xgrid = x.reshape(self.grid[0], self.grid[1])
        ygrid = y.reshape(self.grid[0], self.grid[1])
        zgrid = z.reshape(self.grid[0], self.grid[1])
        
        if(axtype == 'wireframe'): ax.plot_wireframe(x, y, z)
        elif(axtype == 'contour'): ax.contour3D(xgrid, ygrid, zgrid)
        elif(axtype == 'contourf'): ax.contourf3D(xgrid, ygrid, zgrid)
        plt.show()
        
            
            
        
