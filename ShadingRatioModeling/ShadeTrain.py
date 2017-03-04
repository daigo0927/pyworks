import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb

from tqdm import tqdm

from ShadeInit import ShadeInit
from ShadeModel import uGMModel

class ShadeTrain:

    def __init__(self,
                 data_batch,
                 LearningRate = 100.,
                 iterate = 50,
                 mixture = 20):

        self.f = data_batch
        
        self.frame_len = data_batch.shape[0]
        self.frame = np.arange(self.frame_len)

        self.std_frame = np.int((self.frame_len+1)/2)
    
        self.y_len = data_batch.shape[1]
        self.ygrid = np.arange(self.y_len)
    
        self.x_len = data_batch.shape[2]
        self.xgrid = np.arange(self.x_len)

        self.lr = LearningRate

        self.mix = np.int(mixture)

        self.params = None

        self.initmodel = None
        self.model = None

        self.a = None
        self.b = None

        self.loss = []
        self.it = iterate

    def train(self):

        # at first, learn standard frame
        # mean, cov, and pi
        self.initmodel = ShadeInit(obj_data = self.f[self.std_frame])
        
        std_params =  self.initmodel.NormApprox(n_comp = self.mix)

        print('initial learning finished', end='\n')
        
        self.a = self.initmodel.a
        self.b = self.initmodel.b

        # learn movement
        self.model = uGMModel(xy_lim = np.array([self.x_len, self.y_len]),
                              mixture_size = self.mix,
                              frame_num = self.frame_len,
                              logistic_coefficient = np.array([self.a, self.b]))

        self.model.params['mus'] = std_params['mus']
        self.model.params['covs'] = std_params['covs']
        self.model.params['pi'] = std_params['pi']

        for i in tqdm(range(self.it)):
            
            grad_tmp = self.model.gradient_move(f = self.f)
            grad_move = np.mean(grad_tmp, axis = (0, 1, 2))
            self.model.params['move'] -= self.lr * grad_move
    
            self.loss.append(np.mean(self.model.lossvalue))
        
        plt.plot(range(len(self.loss)), self.loss)
        plt.show()

    def plot(self, update = False):
        
        if(update == True):
            self.train()

        plt.figure(figsize = (11, 4*+self.frame_len))
        
        for frm in self.frame:
            
            plt.subplot(self.frame_len, 2, frm*2+1)
            plt.title('original shade ratio')
            sns.heatmap(self.f[frm], annot = False, cmap = 'YlGnBu_r',
                        vmin = 0, vmax = 1)

            plt.subplot(self.frame_len, 2, frm*2+2)
            plt.title('approxed shade ratio')
            sns.heatmap(self.model.g[frm], annot = False, cmap = 'YlGnBu_r',
                        vmin = 0, vmax = 1)

        sns.plt.show()
        
