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
from collections import OrderedDict
import pdb # pdb.set_trace()



class uEpaMMNet(object):

    def __init__(self,
                 grid_size = np.array([50, 50]),
                 input_lim = np.array([10, 10]),
                 mixture_size = 20,
                 frame_num = 5,
                 logistic_coefficient = np.array([50, 0])):

        self.dimension = input_lim.shape[0]
        self.mix = mixture_size
        self.grid = grid_size

        self.frame = np.arange(frame_num)

        xgrid = np.linspace(0, input_lim[0], grid_size[0])
        ygrid = np.linspace(0, input_lim[1], grid_size[1])
        xyt = np.empty((0, self.dimension+1), float)
        for x in xgrid:
            for y in ygrid:
                for f in self.frame:
                    xyt = np.append(xyt, np.array([[x, y, f]]), axis = 0)        
        self.xyt = xyt


        # initilalize parameter
        self.params = {}
        self.params['mus'] = \
                    np.random.rand(mixture_size, self.dimension)*(input_lim)
        tmps = np.array([np.identity(self.dimension)*input_lim/1 \
                         for i in range(self.mix)])
        self.params['covs'] = tmps
        self.params['pi'] = np.random.dirichlet([3]*self.mix)
        move = np.random.rand(self.mix, self.dimension)-0.5
        self.params['move'] = move * input_lim / 10

        self.Epas = []

        self.layers = OrderedDict()
        self.lastLayer = SigmoidWithBregmanDiv(logistic_params = logistic_coefficient)
        
        self.a, self.b = logistic_coefficient

        
    def DistributionUpdate(self):
        
        self.Epas = []
        for f in self.frame:
            mus = self.params['mus'] + self.params['move'] * f
            Epas = [Epanechnikov2Dfunc(mean = mu, cov = cov) \
                    for mu, cov in zip(mus, self.params['covs'])]

            self.Epas.append(Epas)

    def LayerBuild(self):

        self.DistributionUpdate()
        # layer construction

        self.layers['Epanechnikov'] = \
                        EpanechnikovLayer(Epas = self.Epas,
                                          mus = self.params['mus'],
                                          covs = self.params['covs'],
                                          move = self.params['move'])
        self.layers['ReLU'] = ReLU()
        self.layers['Weight'] = Weight(pi = self.params['pi'])
        
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, f):
        q = self.predict(x)
        return self.lastLayer.forward(q, f)

    def GradientCheck(self, x, f):
        loss_W = lambda W: self.loss(x, f)

        grads = {}
        for key in self.params.keys():
            grads[key] = numerical_gradient(loss_W, self.params[key])
        return grads

    def Gradient(self, x, f):
        # forward
        self.loss(x, f)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['mus'] = self.layers['Epanechnikov'].dmus
        grads['covs'] = self.layers['Epanechnikov'].dcovs
        grads['move'] = self.layers['Epanechnikov'].dmove
        grads['pi'] = self.layers['Weight'].dpi

        return grads

    def Plot(self, frame = np.arange(5), axtype = 'contourf'):
        q = np.array([self.predict(x) \
                      for x in self.xyt])

        frame_mask = np.array([self.xyt[:, 2] == f \
                               for f in self.frame])
        x = self.xyt[frame_mask[0], 0]
        y = self.xyt[frame_mask[0], 1]
        Z = sigmoid(self.a * q + self.b)
                                        
        xgrid = x.reshape(self.grid[0], self.grid[1])
        ygrid = y.reshape(self.grid[0], self.grid[1])
        
        for f in frame:
            fig = plt.figure()
            ax = Axes3D(fig)
            z = Z[frame_mask[f]]
            zgrid = z.reshape(self.grid[0], self.grid[1])
            if(axtype == 'wireframe'): ax.plot_wireframe(x, y, z)
            elif(axtype == 'contour'): ax.contour3D(xgrid, ygrid, zgrid)
            elif(axtype == 'contourf'): ax.contourf3D(xgrid, ygrid, zgrid)
            plt.show()

        
            
        

class EpanechnikovLayer:
    
    def __init__(self, Epas, mus, covs, move):
        self.Epas = Epas

        self.mus = mus
        self.covs = covs
        self.move = move

        self.x = None
        self.t = None
        
        self.dmus = None
        self.dcovs = None
        self.dmove = None

    def forward(self, x):
        self.x = x[:2]
        self.t = x[2]
        # pdb.set_trace()
        out = np.array([Epa.value(x = self.x) \
                        for Epa in self.Epas[self.t.astype(np.int64)]])

        

        return out

    def backward(self, dout):
        dx = None

        x_mu_ta = self.x - (self.mus + self.move * self.t)

        # pdb.set_trace()
        self.dmus = np.array([2 * np.linalg.solve(cov, xmt.T) * d \
                              for cov, xmt, d in zip(self.covs, x_mu_ta, dout)])

        
        dcovs_inv = np.array([ -np.dot(np.dot(xmt.T, d).reshape(xmt.shape[0],1),
                                       xmt.reshape(1, xmt.shape[0])) \
                               for xmt, d in zip(x_mu_ta, dout) ])

        dcovs_inv -= np.eye(2,2)

        self.dcovs = np.linalg.inv(dcovs_inv) + np.eye(2,2)

        self.dmove = self.t * self.dmus

        return dx

class ReLU:
    
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # pdb.set_trace()
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Weight:
    
    def __init__(self, pi):
        self.pi = pi
        self.x = None
        self.dpi = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.pi, x.T)

        return out

    def backward(self, dout):
        dx = np.dot(self.pi.T, dout)

        self.dpi = self.x.T * dout

        return dx

class SigmoidWithBregmanDiv:
    
    def __init__(self, logistic_params):
        self.loss = None
        self.g = None
        self.f = None
        self.a, self.b = logistic_params
        self.z = None

    def forward(self, x, f):
        self.f = f
        
        self.z = self.a * x + self.b
        U = 1/self.a * np.log(1 + np.exp(self.z))
        self.loss = U - f * x

        self.g = sigmoid(self.z)

        return self.loss

    def backward(self, dout = 1):
        # batch_size = self.f.shape[0]
        batch_size = 1
        dx = (self.g - self.f)/batch_size

        return dx
        
