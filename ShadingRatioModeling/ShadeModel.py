import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        # ! set basis frame : (frame_num-1)/2 : center frame
        self.std_frame = (frame_num-1)/2

        self.params = {}
        # ! set basis frame : (frame_num-1)/2 : center frame
        self.params['mus'] = np.random.rand(self.mix, self.dimension)\
                             * xy_lim
        self.params['covs'] = np.array([np.identity(self.dimension) * xy_lim/5.
                                        for i in range(self.mix)])
        self.params['pi'] = np.random.dirichlet([3]*self.mix)
        self.params['move'] = (np.random.rand(self.mix, self.dimension) - 0.5) \
                              * xy_lim / 10
        
        self.logistic_coefficient = logistic_coefficient

        self.Norms = None

        self.q_each = None # each component density value
        self.q = None # component mixture value
        self.g = None # infered shade ratio

        self.lossvalue = None


    def predict(self):

        mus_plus = np.array([self.params['mus'] \
                             + self.params['move'] * (f - self.std_frame) \
                             for f in self.frame])
        
        self.Norms = [Norm2Dmix(mus = mus_p,
                                covs = self.params['covs'],
                                pi = self.params['pi'])
                      for mus_p in mus_plus]

        self.q_each = np.array([[[self.Norms[f].pdf_each(x = np.array([x, y]))
                                  for x in self.xgrid]
                                 for y in self.ygrid]
                                for f in self.frame])
        
        self.q = np.sum(self.q_each \
                        * self.params['pi'].reshape(1,1,1,self.mix),
                        axis = 3)

        a, b = self.logistic_coefficient
        z = a * self.q + b
        self.g = sigmoid(z)

    def GenerateFrame(self, f):

        mus_p = self.params['mus'] \
                + self.params['move'] * (f - self.std_frame)

        Norm = Norm2Dmix(mus = mus_p,
                         covs = self.params['covs'],
                         pi = self.params['pi'])

        q = np.array([[Norm.pdf(x = np.array([x, y]))
                       for x in self.xgrid]
                      for y in self.ygrid])

        a, b = self.logistic_coefficient
        z = a * q + b
        g = sigmoid(z)

        # pdb.set_trace()

        return g
        

    def loss(self, f): # f : data value (not frame)
        self.predict()

        a ,b = self.logistic_coefficient
        z = a * self.q + b
        U_q = 1/a * np.log(1 + np.exp(z)) # U function : integrated logistic sigmoid

        self.lossvalue = U_q - f * self.q

    def gradient_move(self, f):
        self.loss(f = f)

        mus_plus = np.array([self.params['mus'] \
                             + self.params['move'] * (frm - self.std_frame) \
                             for frm in self.frame])

        # shape(frame, ygrid, xgrid, mix, 2)
        grad_move = np.array([[[ self.params['pi'].reshape(self.mix, 1) \
                                 * (self.g[frm, y, x] - f[frm, y, x]) \
                                 * self.q_each[frm, y, x, :].reshape(self.mix, 1) \
                                 * (frm - self.std_frame) \
                                 * np.linalg.solve(self.params['covs'],
                                                   (np.array([x, y]) - mus_plus[frm])) \
                                 for x in self.xgrid]
                               for y in self.ygrid]
                              for frm in self.frame])
        return grad_move

    def modelplot(self, update = False):
        if(update == True):
            self.predict()

        plt.figure(figsize = (10, 3.5*(1+self.frame.size)))
        
        for frm in self.frame:
            
            plt.subplot(self.frame.size+1, 2, frm*2+1)
            plt.title('approxed shade ratio')
            sns.heatmap(self.g[frm], annot = False, cmap = 'YlGnBu_r',
                        vmin = 0, vmax = 1)

            plt.subplot(self.frame.size+1, 2, frm*2+2)
            plt.title('approxed probability density')
            sns.heatmap(self.q[frm], annot = False, cmap = 'YlGnBu_r')

        sns.plt.show()


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
        
        
        
            
        
