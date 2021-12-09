# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:06:59 2021

@author: Administrator
"""
import numpy as np
import torch
import torch.nn as nn
from Config import Config
import os
import random
import torch.optim as optim
from my_rician_channel2 import *
from km import *
from operators import *
from generate_loaction import *
from random import sample
import scipy.io  as scio 
import time
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



class ThetaModel(nn.Module):
    def __init__(self,M):
        super(ThetaModel,self).__init__()
        self.M = M
        self.theta = nn.Parameter(torch.rand(2,M)*np.pi*2)

        
    def forward(self,h,gi,gj):        
        ThetaRe = torch.diag(torch.cos(self.theta[0,:]))
        ThetaIm = torch.diag(torch.sin(self.theta[1,:]))
        Theta = torch.cat((ThetaRe.unsqueeze(0),ThetaIm.unsqueeze(0)),0)
        htilde = self.compute_effective_channel(h,gi,gj,Theta) 
        return htilde
    
    def getThetaValue(self):
        ThetaRe = torch.diag(torch.cos(self.theta[0,:]))
        ThetaIm = torch.diag(torch.sin(self.theta[1,:]))
        Theta = torch.cat((ThetaRe.unsqueeze(0),ThetaIm.unsqueeze(0)),0)
        return Theta.detach()

    def _getThetaValue(self):
        ThetaRe = torch.diag(torch.cos(self.theta[0,:]))
        ThetaIm = torch.diag(torch.sin(self.theta[1,:]))
        Theta = torch.cat((ThetaRe.unsqueeze(0),ThetaIm.unsqueeze(0)),0)
        return Theta
    
    def compute_effective_channel(self,h,gi,gj,Theta):
        htilde = torch.zeros_like(h)
        [t1,t2,t3,I,J] = htilde.size()
        for j in range(J):
            for i in range(I):
                htilde[:,:,:,i,j] = h[:,:,:,i,j] + cmul(cmul(gj[:,:,:,j],Theta),gi[:,:,:,i])
        return htilde.squeeze(1).squeeze(1)

class MISSCA(object):
    def __init__(self,config):
        """parameters copied from config"""
        self.config = config
        self.dtype = config.dtype
        self.device = config.device
        
        self.I = config.I #busy user number
        self.J = config.J #idle user number
        self.learning_rate = config.learning_rate
        
        self.As = config.As #slot number in a frame
        self.Af = config.Af #frame number in a time block
        self.M = config.M #number of reflecting elements 
    
        
        self.w = config.w
        self.fi = config.fi
        
        self.pi = config.pi

        self.B = config.B
        self.noise = config.noise
        
        self.Ci = config.Ci
        self.sigma = config.sigma
        self.scale_factor = config.scale_factor
        
        """some parameters that are generated randomly: fj and Li"""
        self.fj = config.fj
        self.Li = config.Li
        
        """network parameter"""
        self.ThetaNN = ThetaModel(self.M)
        self.optimizer = optim.SGD(self.ThetaNN.parameters(),lr=0.001)
        
        """some intermiddle variable"""
        self.rate_matrix = torch.zeros(self.I,self.J)
        self.delay_matrix = torch.zeros(self.I,self.J)
        self.offloading_ratio_matrix = torch.zeros(self.I,self.J)
        self.matching_vector = np.array([0,5,2,4])
        self.delay_recorder = torch.zeros(self.Af,self.As)
        
    def compute_rate(self,htilde):
        self.rate_matrix = torch.zeros(self.I,self.J)
        hnorm = torch.square(htilde[0,:,:])+torch.square(htilde[1,:,:])
        pi = torch.tensor(self.pi).to(dtype=self.dtype,device=self.device)
        pihnorm = torch.mm(torch.diag(pi),hnorm)
        sum_i_power = torch.sum(pihnorm,0)
        
        for i in range(self.I):
            for j in range(self.J):
                self.rate_matrix[i,j] = self.B*torch.log2(1+pihnorm[i,j]/(sum_i_power[j]-pihnorm[i,j]+self.sigma**2))
                #self.rate_matrix[i,j] = self.B*torch.log2(1+pihnorm[i,j]/(self.sigma**2))
                
    def compute_offloading_ratio_and_delay(self):
        self.delay_matrix = torch.zeros(self.I,self.J)
        self.offloading_ratio_matrix = torch.zeros(self.I,self.J)
        for i in range(self.I):
            for j in range(self.J):
                self.offloading_ratio_matrix[i,j] = self.Ci[i]*self.fj[j]*self.rate_matrix[i,j]/(self.Ci[i]*(self.fj[j]+self.fi[i])*self.rate_matrix[i,j]+self.fj[j]*self.fi[i])
                self.delay_matrix[i,j] = (1-self.offloading_ratio_matrix[i,j])*self.Ci[i]*self.Li[i]/self.fi[i]
    
    def compute_matching_strategy(self):
        np_delay_matrix = -self.delay_matrix.detach().numpy()
        delay_matrix = convert_input_format(np_delay_matrix)
        self.matching_vector = run_kuhn_munkres(delay_matrix)
    
    def compute_final_delay(self):
        final_delay = 0
        for i in range(self.I):
            final_delay = final_delay + self.delay_matrix[i,int(self.matching_vector[i])]
            
        return final_delay
        
    def run_MISSCA(self,channel_generator):
        """generate training channels"""
        i=0
        rho_t = 0
        gamma_t = 0
        varpi = 0.4
        theta_bar_t = torch.zeros(2,self.M)
        
        for ix_Af in range(self.Af):
            print('frame number is ====')
            print(ix_Af)
            """test delay performance"""
            h_set,gi_set,gj_set = channel_generator.produce_data_set(self.As,self.config)
            dataSet = list(zip(h_set,gi_set,gj_set))
            self.ThetaNN.eval()
            ix_As = 0
            for h,gi,gj in dataSet:
                htilde = self.ThetaNN(h,gi,gj).detach()
                self.compute_rate(htilde)
                self.compute_offloading_ratio_and_delay() 

                self.compute_matching_strategy()
                self.delay_recorder[ix_Af,ix_As] = self.compute_final_delay()
                ix_As += 1
            print(self.rate_matrix)
            print(self.offloading_ratio_matrix)
            print(self.delay_matrix)
            print(self.matching_vector)
                
            print('average delay is =====')
            print(torch.mean(self.delay_recorder[ix_Af,:]))
            """collect a sample and update phase shifters"""
            self.ThetaNN.train()
            h_set,gi_set,gj_set = channel_generator.produce_data_set(1,self.config)
            dataSet = list(zip(h_set,gi_set,gj_set))
            loss = 0
            for h,gi,gj in dataSet: 
                htilde = self.ThetaNN(h,gi,gj)
                self.compute_rate(htilde)
                self.compute_offloading_ratio_and_delay()
                self.compute_matching_strategy()
                loss = self.compute_final_delay()
                    
            self.optimizer.zero_grad()
            loss.backward()                
            for name, parms in self.ThetaNN.named_parameters():
                if parms.grad is not None:
                    if name == "theta":                       
                        rho_t = (15/(i+15))**(0.6) 
                        gamma_t = 10/(10+i)
                        theta_bar_t = (1-rho_t)*theta_bar_t + rho_t*parms.grad
                        parms.data = parms.data - gamma_t*theta_bar_t/(2*varpi)
                        print("grad value is")
                        print(torch.mean(torch.abs(parms.grad)))
            i=i+1
            
            
            

           
if __name__ == '__main__':
    seed = 20210517
    torch.manual_seed(seed)  
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    
    config = Config()
    location = LocationSetting(config)
    
    position = location.getLocation()    
    plt.figure()    
    [position_IRS,position_up_users,position_down_users] = position
    for i in range(config.I):
        plt.scatter(position_up_users[0,i], position_up_users[1,i],c='r')       
    for j in range(config.J):
        plt.scatter(position_down_users[0,j], position_down_users[1,j],c='g')
        
    channel_generator =  My_rician_channel(config.I, config.J, 1, 1, config.M, position)
    
    missca = MISSCA(config)
    missca.run_MISSCA(channel_generator)

    
    plt.figure()
    xAxis = np.linspace(1,config.Af,config.Af)
    yAxis = torch.mean(missca.delay_recorder,1).detach().numpy()
    plt.plot(xAxis,yAxis)
                
            
        