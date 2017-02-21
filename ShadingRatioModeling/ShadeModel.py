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

class uEpaMixModel(object):

    def __init__(self,
                 grid_size = np.array([50, 50]),
                 input_lim = np.array([10, 10]),
                 mixture_size = 20,
                 frame_num = 5,
                 logistic_coefficient = np.array([50, 0])):

        self.dimension = input_lim.shape[0]
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

        self.params = {}
        self.params['mus'] = \
                    np.random.rand(mixture_size, self.dimension)*(input_lim)
        tmps = np.array([np.identity(self.dimension)*input_lim/10 \
               for i in range(self.mix)])
        self.params['covs'] = tmps
        self.params['pi'] = np.random.dirichlet([3]*self.mix)
        move = np.random.rand(self.mix, self.dimension)-0.5
        self.params['move'] = move * input_lim / 10

        self.mus_plus = None
        
        self.logistic_coefficient = logistic_coefficient

        self.predict_value = None
        
        self.Epas = None
        
        

    def predict(self):
        
        xy = self.xy

        mus = self.params['mus']
        covs = self.params['covs']
        pi = self.params['pi']

        move = self.params['move']
        self.mus_plus = np.array([(mus + move*frame) \
                                  for frame in self.frame])

        a, b = self.logistic_coefficient

        self.Epas = [Epanechnikov2Dmix(mus = mus_p, covs= covs, pi=pi) \
                     for mus_p in self.mus_plus]

        q =  np.array([self.Epas[frame].pdf(xy) \
                       for frame in self.frame])
        g = sigmoid(a * q + b)

        return g
    
    def Gradient(self, f):

        # prediction for each frames
        self.predict_value = self.predict()

        # material for gradient ----------
        diff_gf = self.predict_value - f # shape(frame, grid)
        diff_gf = diff_gf.reshape(1, diff_gf.shape[0], diff_gf.shape[1]) # shape(1,frame, grid)
        diff_gf = np.tile(diff_gf, (self.mix, 1,1))
        # shape(mix, frame, grid)
        diff_gf_vec = diff_gf.reshape(diff_gf.shape[0],
                                      diff_gf.shape[1],
                                      1,
                                      diff_gf.shape[2])
        diff_gf_vec = np.tile(diff_gf_vec, (1,1,2,1))
        # shape(mix, frame, 2, grid)
        diff_gf_mat = diff_gf.reshape(diff_gf.shape[0],
                                      diff_gf.shape[1],
                                      1,
                                      1,
                                      diff_gf.shape[2])
        diff_gf_mat = np.tile(diff_gf_mat, (1,1,2,2,1))
        # shape(mix, frame, 2, 2, grid)

        pdf_each_value = np.array([self.Epas[frame].pdf_each_value \
                                   for frame in self.frame]) # shape(frame, mix, grid)
        pdf_each_value = np.transpose(pdf_each_value, (1,0,2))
        # shape(mix, frame, grid)
        
        pdf_each_mask = pdf_each_value>0
        # shape(mix, frame, grid)
        pdf_each_mask_vec = pdf_each_mask.reshape(pdf_each_mask.shape[0],
                                                  pdf_each_mask.shape[1],
                                                  1,
                                                  pdf_each_mask.shape[2])
        pdf_each_mask_vec = np.tile(pdf_each_mask_vec, (1,1,2,1))
        # shape(mix, frame, 2, grid)
        pdf_each_mask_mat = pdf_each_mask.reshape(pdf_each_mask.shape[0],
                                                  pdf_each_mask.shape[1],
                                                  1,
                                                  1,
                                                  pdf_each_mask.shape[2])
        pdf_each_mask_mat = np.tile(pdf_each_mask_mat, (1,1,2,2,1))
        # shape(mix, frame, 2, 2, grid)

        mu_ta = np.transpose(self.mus_plus, (1, 0, 2)) # shape(mix, frame, 2)
        mu_ta = mu_ta.reshape(mu_ta.shape[0], mu_ta.shape[1], mu_ta.shape[2], 1)
        # shape(mix, frame, 2, 1)
        x = self.xy.T.reshape(1,1,2, self.xy.shape[0]) # shape(1,1,2,grid)
        x = np.tile(x,
                    (self.mix, self.frame.shape[0], 1, 1)) # shape(mix, frame, 2, grid)
        x_mu_ta = x - mu_ta
        # shape(mix, frame, 2, grid)
        
        tmp1 = x_mu_ta.reshape(x_mu_ta.shape[0],
                               x_mu_ta.shape[1],
                               x_mu_ta.shape[2],
                               1,
                               x_mu_ta.shape[3] # shape(mix, frame, 2,1, grid)
                               )
        tmp2 = x_mu_ta.reshape(x_mu_ta.shape[0],
                               x_mu_ta.shape[1],
                               1,
                               x_mu_ta.shape[2],
                               x_mu_ta.shape[3]) # shape(mix, frame, 1,2, grid)
        x_mu_ta_twice = np.zeros(shape = (self.mix,
                                          self.frame.shape[0],
                                          2,2,
                                          x_mu_ta.shape[3])) # shape(mix, frame, 2,2, gird)
        for m in range(self.mix):
            for f in self.frame:
                for g in range(x_mu_ta.shape[3]):
                    x_mu_ta_twice[m, f, :, :, g] = \
                                    np.dot(tmp1[m, f, :, :, g],
                                           tmp2[m, f, :, :, g])
        # x_mu_ta_twice shape(mix, frame, 2, 2, grid)
        
        
        pi_for_vec = self.params['pi'].reshape(self.mix, 1,1,1)
        pi_for_mat = self.params['pi'].reshape(self.mix, 1,1,1,1)

        covs = self.params['covs']
        covs = covs.reshape(covs.shape[0],
                            1,
                            covs.shape[1],
                            covs.shape[2])
        covs = np.tile(covs, (1, self.frame.shape[0], 1, 1))
        # shape(mix, frame, 2, 2)

        t = self.frame.reshape(1, self.frame.shape[0], 1, 1)
        t = np.tile(t, (self.mix, 1,1,1))
        # shape(mix, frame, 1,1)
        
        grads = {}
        grads['pi'] = np.sum(diff_gf * pdf_each_value, axis = (1,2))
        # shape(mix)
        grads['mus'] = np.sum(pi_for_vec * diff_gf_vec \
                              * (2 * np.linalg.solve(covs, x_mu_ta) \
                                 * pdf_each_mask_vec), # shape(mix, frame, 2, grid)
                              axis = (1,3))
        # shape(mix, 2)
        grad_covs_inv = np.sum(pi_for_mat * diff_gf_mat \
                               * (- x_mu_ta_twice) * pdf_each_mask_mat,
                               # shape(mix, frame, 2, 2, grid)
                               axis = (1, 4))
        grads['covs'] = np.linalg.inv(grad_covs_inv)
        # shape(mix, 2, 2)

        grads['move'] = np.sum(pi_for_vec * diff_gf_vec \
                               * (2 * t * np.linalg.solve(covs, x_mu_ta) \
                                  * pdf_each_mask_vec), # shape(mix, frame, 2, grid)
                               axis = (1, 3))
        # shape(mix, 2)
        
        return(grads)

        
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
        
        
        
            
        
