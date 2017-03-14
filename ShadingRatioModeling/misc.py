import sys, os
sys.path.append(os.pardir)
import numpy as np
import itertools
from common.functions import sigmoid
from common.gradient import numerical_gradient
from PIL import Image
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pdb


def sigmoid(z):
    return(1/(1+np.exp(-z)))

class Norm2Dmix:

    def __init__(self, mus, covs, pi):
        self.mus = mus
        self.covs = covs
        self.pi = pi

        self.Norms = [multivariate_normal(mean = mu, cov = cov) \
                      for mu, cov in zip(self.mus, self.covs)]

    def pdf(self, x):
        q = np.array([Norm.pdf(x = x) \
                      for Norm in self.Norms])
        q_WeightedSum = np.sum(q * self.pi)

        return q_WeightedSum

    def pdf_each(self, x):
        q = np.array([Norm.pdf(x = x) \
                      for Norm in self.Norms])
        return q

def Time2Str(year = 2016,
             month = 1,
             day = 1,
             hour = 12,
             minute = 0,
             second = 0,
             sep = '-'):

    year_str = str(year)

    if month < 10:
        month_str = '0' + str(month)
    else:
        month_str = str(month)

    if day < 10:
        day_str = '0' + str(day)
    else:
        day_str = str(day)

    if hour < 10:
        h_str = '0' + str(hour)
    else:
        h_str = str(hour)

    if minute < 10:
        m_str = '0' + str(minute)
    else:
        m_str = str(minute)
        
    if second < 10:
        s_str = '0' + str(second)
    else:
        s_str = str(second)

    timestr = sep.join([year_str, month_str, h_str, m_str, s_str])

    return timestr

def Time2Strings(year = 2016,
                 month = 8,
                 day = 20,
                 start_hour = 9,
                 start_minute = 0,
                 start_second = 0,
                 end_hour = 12,
                 end_minute = 0,
                 end_second = 0,
                 grid_second = 10,
                 sep = '-'):

    h = start_hour
    m = start_minute
    s = start_second

    TimeStrList = []
    status = True

    while status:
        timestring = Time2Str(year = year,
                              month = month,
                              day = day,
                              hour = h,
                              minute = m,
                              second = s,
                              sep = sep)
        
        TimeStrList.append(timestring)
        
        s = s + grid_second
        m = m + s//60
        s = s%60

        h = h + m//60
        m = m%60

        # pdb.set_trace()

        if(h == end_hour and m == end_minute and s > end_second):
            status = False

            

    return TimeStrList



        
    
class Epanechnikov2Dfunc:

    ## return without 0-cut value
    # 2 variate Epanechnikov function
    # f(x) = max(0, D-(x-mean)*cov*(x-mean.T))
    # D : normalize constant
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

        h = 1e-4
        a = cov[0,0]
        b = cov[0,1]
        c = cov[1,1]
        if(a == c):
            c = c+h

        theta = 1/2*np.arctan(2*b/(a-c))
        tmp1 = \
            c*np.cos(theta)**2 - 2*b*np.sin(theta)*np.cos(theta) + a*np.sin(theta)**2
        tmp2 = \
            c*np.sin(theta)**2 + 2*b*np.sin(theta)*np.cos(theta) + a*np.cos(theta)**2
        det = a*c-b**2
        self.D = np.sqrt(2 * np.sqrt(tmp1*tmp2)/np.pi/det)

    def value(self, x):
        mean = self.mean
        cov = self.cov
        D = self.D

        value = D - np.dot((x-mean), np.linalg.solve(cov, (x-mean).T))

        return value


class Epanechnikov2D:

    # 2 variate Epanechnikov function
    # f(x) = max(0, D-(x-mean)*cov*(x-mean.T))
    # D : normalize constant
    def __init__(self, mean, cov):
        self.Epa = Epanechnikov2Dfunc(mean = mean, cov = cov)

    def pdf(self, x):

        density = max(0, self.Epa.value(x = x))
        
        return density, density>0

class Epanechnikov2Dmix:
    
    def __init__(self, mus, covs, pi):
        self.mus = mus
        self.covs = covs
        self.pi = pi

        self.Epas = [Epanechnikov2D(mean = mu, cov = cov) \
                     for mu, cov in zip(self.mus, self.covs)]

    def pdf_and_mask(self, x):
        
        q_and_mask = np.array([Epa.pdf(x = x) \
                               for Epa in self.Epas]) # shape(mix, 2)
        q = q_and_mask[:,0]
        mask = q_and_mask[:,1]
        q_weighted_sum = np.sum(q * self.pi)

        # contain density value, and mask at the same time
        # 0-th column : mixture density value
        # 1- column : each component masks
        return np.r_[q_weighted_sum, mask]

    def pdf(self, x):

        q_and_mask = np.array([Epa.pdf(x = x) \
                               for Epa in self.Epas]) # shape(mix, 2)
        q = q_and_mask[:,0]
        q_weighted_sum = np.sum(q * self.pi)

        return q_weighted_sum

    def pdf_each(self, x):
        
        q_and_mask = np.array([Epa.pdf(x = x) \
                               for Epa in self.Epas]) # shape(mix, 2)
        q = q_and_mask[:,0]

        return q
    
