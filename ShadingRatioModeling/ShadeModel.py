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
import pdb # pdb.set_trace()

class uGMModel:

    def __init__(self,
                 xy_lim = np.array([30, 30]),
                 mixture_size = 20,
                 frame_num = 5,
                 logistic_coefficient = np.array([50, 0])):
        
        self.dimension = xy_lim.shape[0]
        self.mix = mixture_size
        
        self.xgrid = np.arange(xy_lim[0])
        self.ygrid = np.arange(xy_lim[1])
        
        self.frame = np.arange(frame_num)

        self.params = {}
        self.params['mus'] = np.random.rand(self.mix, self.dimension)\
                             * xy_lim
        self.params['covs'] = np.array([np.identity(self.dimension) * xy_lim/5.
                                        for i in range(self.mix)])
        self.params['pi'] = np.random.dirichlet([3]*self.mix)
        self.params['move'] = (np.random.rand(self.mix, self.dimension) - 0.5) \
                              * xy_lim / 10
        
        self.logistic_coefficient = logistic_coefficient

        self.Norms = None
        self.Norms_specific_frame = None

    def predict_specific_frame(self, obj_frame):
        self.Norms_specific_frame = Norm2Dmix(mus = self.params['mus'] \
                                              + self.params['move'] * obj_frame,
                                              covs = self.params['covs'],
                                              pi = self.params['pi'])

        q = np.array([[self.Norms_specific_frame.pdf(x = np.array([x, y]))
                       for x in self.xgrid]
                      for y in self.ygrid])

        return q

    def predict(self):

        mus_plus = np.array([self.params['mus'] + self.params['move'] * f \
                             for f in self.frame])
        
        self.Norms = [Norm2Dmix(mus = mus_p,
                                covs = self.params['covs'],
                                pi = self.params['pi'])
                      for mus_p in mus_plus]

        q = np.array([[[self.Norms[f].pdf(x = np.array([x, y]))
                        for x in self.xgrid]
                       for y in self.ygrid]
                      for f in self.frame])

        return q

    def loss_specific_frame(self, f, obj_frame):
        q = self.predict_specific_frame(obj_frame = obj_frame)

        a, b = self.logistic_coefficient

        z = a * q + b
        U_q = 1/a * np.log(1 + np.exp(z))

        loss = U_q - f[obj_frame] * q

        return loss
    
    def loss(self, f):
        q = self.predict()

        a ,b = self.logistic_coefficient

        z = a * q + b
        U_q = 1/a * np.log(1 + np.exp(z))

        loss = U_q - f * q

        return loss

    


class uEpaMixModel(object):

    def __init__(self,
                 xy_lim = np.array([30, 30]),
                 mixture_size = 20,
                 frame_num = 5,
                 logistic_coefficient = np.array([50, 0])):

        self.dimension = xy_lim.shape[0]
        self.mix = mixture_size

        self.xgrid = np.arange(start = 0, stop = xy_lim[0], step = 1)
        self.ygrid = np.arange(start = 0, stop = xy_lim[1], step = 1)

        self.frame = np.arange(frame_num)

        self.params = {}
        self.params['mus'] = \
                    np.random.rand(self.mix, self.dimension)\
                    *(xy_lim)
        tmp = np.array([np.identity(self.dimension)*xy_lim/0.1 \
                         for i in range(self.mix)])
        self.params['covs'] = tmp
        self.params['pi'] = np.random.dirichlet([3]*self.mix)
        move = np.random.rand(self.mix, self.dimension)-0.5
        self.params['move'] = move * xy_lim / 10

        self.mus_plus = None
        
        self.logistic_coefficient = logistic_coefficient

        self.q = None
        self.q_specific = None

        # self.Epas[frame].Epas[mix]
        self.Epas = None
        self.Epas_specific_frame = None
        

    def predict_specific_frame(self, obj_frame = 0):

        self.Epas_specific_frame = Epanechnikov2Dmix(mus = self.params['mus'] \
                                                     + self.params['move'] * obj_frame,
                                                     covs = self.params['covs'],
                                                     pi = self.params['pi'])

        q_and_mask = np.array([[self.Epas_specific_frame.pdf_and_mask(x = np.array([x, y]))
                                for x in self.xgrid]
                               for y in self.ygrid])
        
        self.q_specific = q_and_mask[:,:,0]
        mask = q_and_mask[:,:,1:]

        # pdb.set_trace()
        
        # return specific frame predict value, and component mask
        # q.shape : (ygird ,xgrid)
        # mask.shape : (ygrid, xgrid, mix)
        return self.q_specific, mask
        

    def predict(self):
        
        self.mus_plus = np.array([self.params['mus'] + self.params['move'] * f \
                                  for f in self.frame])

        self.Epas = [Epanechnikov2Dmix(mus = mus_p,
                                       covs = self.params['covs'],
                                       pi = self.params['pi'])
                     for mus_p in self.mus_plus]

        # compute from given frames
        q_and_mask = np.array([[[self.Epas[f].pdf_and_mask(x = np.array([x, y]))
                                 for x in self.xgrid]
                                for y in self.ygrid]
                               for f in self.frame])
        # pdb.set_trace()

        self.q = q_and_mask[:,:,:,0]
        mask = q_and_mask[:,:,:,1:]

        # return predict value, and component mask
        # q.shape : (frame, ygird ,xgrid)
        # mask.shape : (frame, ygrid, xgrid, mix)
        return self.q, mask

    def loss_specific_frame(self, f, obj_frame = 0):
        
        q, mask = self.predict_specific_frame(obj_frame = obj_frame)

        a, b = self.logistic_coefficient

        z = a * q + b
        U_q = 1/a * np.log(1 + np.exp(z))

        loss = U_q - f[obj_frame] * q

        return loss, mask

    def loss(self, f): # f : data value
        
        q, mask = self.predict()
        
        a, b = self.logistic_coefficient

        z = a * q + b
        U_q = 1/a * np.log(1 + np.exp(z))

        loss = U_q - f * q

        return loss, mask

    def gradient_specific_frame(self, f, obj_frame = 0):

        # it will get better for estimate mus, covs, and pi at first (without move)
        # return gradient of mus, covs, pi
        loss, mask = self.loss_specific_frame(f = f,
                                              obj_frame = obj_frame)

        self.Epas_specific_frame

        mus = self.params['mus'] + self.params['move'] * obj_frame
        covs = self.params['covs']
        
        a, b = self.logistic_coefficient

        z = a * self.q_specific + b
        g = sigmoid(z)

        f = f[obj_frame]

        dpi = np.array([[ (g[y,x] - f[y,x]) \
                          * self.Epas_specific_frame.pdf_each(x = np.array([x, y]))
                          for x in self.xgrid]
                        for y in self.ygrid])

        dmus = np.array([[ self.params['pi'] * (g[y,x] - f[y,x]) * 2 \
                           * np.linalg.solve(covs, (np.array([x, y]) - mus) ).T \
                           * mask[y, x]
                           for x in self.xgrid]
                         for y in self.ygrid])

        # dcovs = np.array([[   ]])
        
        pdb.set_trace()
        
        
        
    
    def gradient(self, f):

        loss, mask = self.loss(f = f)
        
        a, b = self.logistic_coefficient

        z = a * self.q + b
        g = sigmoid(z)

        
        
               
        
    def ModelPlot(self, frame = range(5), axtype='contourf'):

        if(self.predict_value== None): self.predict_value = self.predict()
        
        x = self.xy[:, 0]
        y = self.xy[:, 1]
        Z = self.predict_value
        xgrid = x.reshape(self.grid[0], self.grid[1])
        ygrid = y.reshape(self.grid[0], self.grid[1])
        
        for f in frame:
            fig = plt.figure()
            ax = Axes3D(fig)
            z = Z[f]
            zgrid = z.reshape(self.grid[0], self.grid[1])
            if(axtype == 'wireframe'): ax.plot_wireframe(x, y, z)
            elif(axtype == 'contour'): ax.contour3D(xgrid, ygrid, zgrid)
            elif(axtype == 'contourf'): ax.contourf3D(xgrid, ygrid, zgrid)
            plt.show()
        
        
        
            
        
