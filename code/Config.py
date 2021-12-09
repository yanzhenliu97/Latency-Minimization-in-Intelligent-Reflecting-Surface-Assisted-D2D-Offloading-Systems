# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:54:33 2021

@author: Administrator
"""


import numpy as np
import torch
class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self):
        self.USE_GPU = False
        if self.USE_GPU == False:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')
        self.dtype = torch.float32
        self.I = 4 #busy user number
        self.J = 6 #idle user number
        self.learning_rate = 0.001
        
        self.As = 50 #slot number in a frame
        self.Af = 50 #frame number in a time block
        self.M = 32 #number of reflecting elements 
    
        
        self.w = 1*np.ones(self.I);
        self.fi = 1e9*np.ones(self.I)
        
        self.pi = 24*np.ones(self.I)
        self.pi = 10**((self.pi-30)/10)
        """some parameters that are generated randomly: fj and Li"""
        self.fj = 1e9*1*np.ones(self.J)+4*1e9*np.random.rand(self.J)
        self.Li = 1e6*(np.ones(self.I)+4*np.random.rand(self.I))
        
        self.B = 15*1e6
        self.noise = 10**(-(174+30)/10)*self.B
        
        self.Ci = 100*np.ones(self.I)
        self.sigma = 1
        self.scale_factor = np.sqrt(self.sigma**2/self.noise*self.pi[0])
        
        self.R1 = 10
        self.R2 = 10
        self.O1 = -8
        self.O2 = 8
        self.heighti = 3
        self.heightu = 1