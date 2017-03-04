import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
from sklearn.mixture import GMM
from scipy.interpolate import interp2d
from misc import sigmoid, Norm2Dmix


class ShadeInit:

    def __init__(self, obj_data):
        self.f = obj_data
        self.xgrid = np.arange(obj_data.shape[1])
        self.ygrid = np.arange(obj_data.shape[0])

        self.fouter = None

        self.a = None
        self.b = None

        self.params = None

        self.p = None
        self.pouter = None

    def ComputeLogisticCoef(self, outerpolate = 3):
        fmin = np.min(self.f)

        self.b = np.log(fmin/(1 - fmin))

        # outerpolate original data
        # because original data may divide some component
        interfunc = interp2d(self.xgrid, self.ygrid, self.f, kind = 'linear')

        xouter = np.arange(start = -outerpolate, stop = len(self.xgrid)+outerpolate)
        youter = np.arange(start = -outerpolate, stop = len(self.xgrid)+outerpolate)

        self.fouter = interfunc(xouter, youter)
        
        f = self.fouter + 1e-6
        self.a = np.sum(np.log(f/(1-f)) - self.b)

        return self.a, self.b

    def Samplize(self, sample_num = 1e+5, outerpolate = 3):
        self.ComputeLogisticCoef(outerpolate = outerpolate)

        f = self.fouter + 1e-6

        # convert data value to pdf value
        p = np.log(f/(1-f))/self.a - self.b/self.a
        p = p/np.sum(p)
        self.pouter = p

        p_vec = p.reshape(p.size)

        idx_sample = np.random.choice(range(p.size), np.int(sample_num), p = p_vec)

        xy_sample = np.array([np.array([idx%f.shape[1] - outerpolate,
                                        idx//f.shape[0] - outerpolate])
                              for idx in idx_sample])

        return xy_sample

    def NormApprox(self, sample_num = 1e+5, n_comp = 10):

        sample = self.Samplize(sample_num = sample_num)

        gmm = GMM(n_components = n_comp, covariance_type = 'full')
        gmm.fit(sample)

        self.params = {}
        self.params['mus'] = gmm.means_
        self.params['covs'] = gmm.covars_
        self.params['pi'] = gmm.weights_

        return self.params

    def ApproxPlot(self, update = False):
        if update == True:
            print('input generated sample number')
            s_num = np.int(input())
            print('input component number used for approxed')
            n_comp = np.int(input())
            self.NormApprox(sample_num = s_num, n_comp = n_comp)
            

        Norm = Norm2Dmix(mus = self.params['mus'],
                         covs = self.params['covs'],
                         pi = self.params['pi'])

        q = np.array([[Norm.pdf(x = np.array([x, y]))
                       for x in self.xgrid]
                      for y in self.ygrid])

        g = sigmoid(self.a * q + self.b)

        
        plt.figure(figsize = (12, 8))

        plt.subplot(221)
        plt.title('original shade ratio')
        sns.heatmap(self.f, annot = False, cmap = 'YlGnBu_r', vmin = 0, vmax = 1)

        plt.subplot(222)
        plt.title('approxed shade ratio')
        sns.heatmap(g, annot = False, cmap = 'YlGnBu_r', vmin = 0, vmax = 1)

        plt.subplot(223)
        plt.title('outerpolated shade ratio')
        sns.heatmap(self.fouter, annot = False, cmap = 'YlGnBu_r', vmin = 0, vmax = 1)
        
        plt.subplot(224)
        plt.title('approxed probability density')
        sns.heatmap(q, annot = False, cmap = 'YlGnBu_r')
        
        sns.plt.show()

        

        
        
        
        
