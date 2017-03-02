import numpy as np
import pdb
from sklearn.mixture import GMM

class ShadeInit:

    def __init__(self, obj_data):
        self.f = obj_data
        self.xgrid = np.arange(obj_data.shape[1])
        self.ygrid = np.arange(obj_data.shape[0])

        self.a = None
        self.b = None

    def ComputeLogisticCoef(self):
        fmin = np.min(self.f)

        self.b = np.log(fmin/(1 - fmin))

        f = self.f + 1e-6
        self.a = np.sum(np.log(f/(1-f)) - self.b)

        return self.a, self.b

    def Samplize(self, sample_num = 1e+5):
        f = self.f + 1e-6

        # convert data value to pdf value
        p = np.log(f/(1-f))/self.a - self.b/self.a
        p = p/np.sum(p)

        p_vec = p.reshape(p.size)

        idx_sample = np.random.choice(range(p.size), np.int(sample_num), p = p_vec)

        xy_sample = np.array([np.array([idx%self.f.shape[1], idx//self.f.shape[0]])
                              for idx in idx_sample])

        return xy_sample

    def NormApprox(self):
        
