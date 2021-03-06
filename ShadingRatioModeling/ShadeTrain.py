# coding : utf-8

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb
import itertools as itr

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import Process

from ShadeInit import ShadeInit
from ShadeModel import uGMModel
from misc import Time2Strings



class ShadeSystem:
    
    def __init__(self,
                 data,
                 batch_size = 3,
                 mixture = 20,
                 stride = 1):

        self.batch_size = batch_size
        self.stride = stride

        self.data = data
        self.data_len = data.shape[0]

        self.mix = mixture

        try:
            if (self.data_len - self.batch_size) % self.stride != 0:
                raise ValueError('batchsize and stride must be set appropriately')
            elif (self.batch_size - self.stride) <= 1:
                raise ValueError('you should change the batchsize or stride for good TSR')
        except ValueError as e:
            print(e)
            
        # split the data for training
        self.data_batches = [[data[i:i+self.batch_size], self.mix]
                             for i in np.arange(start = 0,
                                                stop = self.data_len \
                                                - self.batch_size + 1,
                                                step = self.stride)]

        self.FitResult = None

        self.CompleteFrames = None

    def fit(self):
        core_num = np.int(input('input core number : '))
        pool = Pool(core_num)
        
        self.FitResult = list(pool.map(train, self.data_batches))


    def TemporalInterpolate(self, finess = 15):
        
        core_num = np.int(input('input core number : '))
        pool = Pool(core_num)

        modelpairs = [[self.FitResult[i].model, self.FitResult[i+1].model, finess]
                      for i in range(len(self.FitResult) - 1)]

        # shape(pair, finess, ygrid, xgrid)
        middleframes = np.array(pool.map(interp, modelpairs))

        # shape(pair*finess, ygid, xgrid)
        middleframes = middleframes.reshape(middleframes.shape[0] \
                                            * middleframes.shape[1],
                                            middleframes.shape[2],
                                            middleframes.shape[3])

        firstmodel = self.FitResult[0].model
        lastmodel = self.FitResult[-1].model
        firstframes = np.array([firstmodel.GenerateFrame(f = fin)
                                for fin in (np.arange(finess)/finess)])
        lastframes = np.array([lastmodel.GenerateFrame(f = lastmodel.std_frame + fin)
                               for fin in (np.arange(finess+1)/finess)])

        self.CompleteFrames = np.concatenate((firstframes,
                                              middleframes,
                                              lastframes),
                                             axis = 0)
        # pdb.set_trace()

    def SaveInterp(self, path):
        
        start_date = input('input start date ex ) 2016-1-1-12-0-0 : ')
        end_date = input('input end date ex ) 2016-1-1-18-0-0 : ')
        
        s_year, s_month, s_day, s_hour, s_minute, s_second \
            = list(map(int, start_date.split('-')))
        e_year, e_month, e_day, e_hour, e_minute, e_second \
            = list(map(int, end_date.split('-')))

        try:
            if not (s_year == e_year and s_month == e_month and s_day == e_day):
                raise ValueError('can\'t perform across dates')
        except ValueError as e:
            print(e)
            sys.exit()

        TimeStrList = Time2Strings(year = s_year,
                                   month = s_month,
                                   day = s_day,
                                   start_hour = s_hour,
                                   start_minute = s_minute,
                                   start_second = s_second,
                                   end_hour = e_hour,
                                   end_minute = e_minute,
                                   end_second = e_second,
                                   grid_second = 10,
                                   sep = '-')

        PathList = list(map(lambda timestr: path + timestr + '.png',
                            TimeStrList))


        frame_and_path = [[frame, pth]
                          for frame, pth in zip(self.CompleteFrames, PathList)]
        
        core_num = np.int(input('input core number : '))
        pool = Pool(core_num)

        pool.map(SaveImage, frame_and_path)

        
    def SaveParams(self, path):
        
        for i, Tner in enumerate(self.FitResult):
            savepath = '{}/trainer{}.pkl'.format(path, i)
            Tner.save(path = savepath)


def interp(modelpair_and_finess):

    premodel, postmodel, finess = modelpair_and_finess
    
    # finess = 15
    # premodel = modelpair[0]
    # postmodel = modelpair[1]

    synthesis = np.array([premodel.GenerateFrame(f = premodel.std_frame+fin) \
                          * (1 - fin) \
                          + postmodel.GenerateFrame(f = fin) \
                          * fin
                          for fin in (np.arange(finess)/finess)])
    
    return synthesis


def SaveImage(data_and_path):
    
    # (data, path)
    data, path = data_and_path
    # path : path/to/---.png
    date = path.split('/')[-1].split('.')[0] # extract date : '---'

    sns.plt.title(date)
    sns.heatmap(data, vmin = 0, vmax = 1,
                cmap = 'YlGnBu_r', annot = False)
    sns.plt.savefig(path)
    sns.plt.close()
    
    
    
        
def train(data_batch_and_mix):

    data_batch, mix = data_batch_and_mix

    trainer = Trainer(data_batch = data_batch,
                      outer_drop = 5,
                      LearningRate = 100,
                      iterate = 50,
                      mixture = mix)
    
    trainer.train(plot = False)

    return trainer


class Trainer:

    def __init__(self,
                 data_batch,
                 outer_drop = 5,
                 LearningRate = 100.,
                 iterate = 50,
                 mixture = 20):

        self.f = data_batch

        # drop outer area for catching large scale cloud move
        self.outer_drop = outer_drop 
        
        self.frame_len = data_batch.shape[0]
        self.frame = np.arange(self.frame_len)

        self.std_frame = np.int((self.frame_len+1)/2)

        # drop outer area
        self.y_len = data_batch.shape[1] - 2*self.outer_drop
        self.ygrid = np.arange(self.y_len)

        # drop outer area
        self.x_len = data_batch.shape[2] - 2*self.outer_drop
        self.xgrid = np.arange(self.x_len)

        # objective area (outer area dropped)
        self.f_objective = data_batch[:,
                                      self.outer_drop : self.outer_drop+self.y_len,
                                      self.outer_drop : self.outer_drop+self.x_len]

        self.lr = LearningRate

        self.mix = np.int(mixture)

        self.params = None

        self.initmodel = None
        self.model = None

        self.a = None
        self.b = None

        self.loss = []
        self.it = iterate

    def train(self, plot = True):

        # at first, learn standard frame
        # mean, cov, and pi
        self.initmodel = ShadeInit(obj_data = self.f[self.std_frame])
        
        std_params =  self.initmodel.NormApprox(n_comp = self.mix,
                                                outerpolate = 0)

        print('initial learning finished')
        
        self.a = self.initmodel.a
        self.b = self.initmodel.b

        # learn movement
        self.model = uGMModel(xy_lim = np.array([self.x_len, self.y_len]),
                              mixture_size = self.mix,
                              frame_num = self.frame_len,
                              logistic_coefficient = np.array([self.a, self.b]))

        self.model.params['mus'] = std_params['mus'] - self.outer_drop
        self.model.params['covs'] = std_params['covs']
        self.model.params['pi'] = std_params['pi']

        
        for i in tqdm(range(self.it)):
            
            # shape(frame, y, x, 2)
            grad_tmp = self.model.gradient_move(f = self.f_objective) 
            grad_move = np.mean(grad_tmp, axis = (0, 1, 2))
            self.model.params['move'] -= self.lr * grad_move
            
            self.loss.append(np.mean(self.model.lossvalue))

        if plot:
            plt.plot(range(len(self.loss)), self.loss)
            plt.show()

    def plot(self, update = False):
        
        if(update == True):
            self.train()

        plt.figure(figsize = (11, 4*+self.frame_len))
        
        for frm in self.frame:
            
            plt.subplot(self.frame_len, 2, frm*2+1)
            plt.title('original shade ratio')
            sns.heatmap(self.f_objective[frm], annot = False, cmap = 'YlGnBu_r',
                        vmin = 0, vmax = 1)

            plt.subplot(self.frame_len, 2, frm*2+2)
            plt.title('approxed shade ratio')
            sns.heatmap(self.model.g[frm], annot = False, cmap = 'YlGnBu_r',
                        vmin = 0, vmax = 1)

        sns.plt.show()

    def save(self, path):
        
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)    
    
    def load(self, path):

        with open(path, 'rb') as f:
            contents = pickle.load(f)

        for key in contents.keys():
            self.__dict__[key] = contents[key]

        
        
