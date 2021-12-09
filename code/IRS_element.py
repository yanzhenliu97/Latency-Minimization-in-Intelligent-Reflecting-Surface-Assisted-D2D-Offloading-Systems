# -*- coding: utf-8 -*-
"""
Created on Wed May 19 15:21:46 2021

@author: yanzhenliu
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
from MISSCA import ThetaModel,MISSCA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class MISSCA_performance(MISSCA):
    def __init__(self,config,whether_heuristic_matching = False):
        super(MISSCA_performance,self).__init__(config)
        
        self.whether_heuristic_matching = whether_heuristic_matching
        if whether_heuristic_matching:
            self.find_heuristic_matching_strategy()
    
    def find_heuristic_matching_strategy(self):
        Li_index = np.argsort(-self.Li)
        fj_index = np.argsort(-self.fj)
        for i in range(self.I):
            self.matching_vector[Li_index[i]] = fj_index[i]
        
        
    def run_MISSCA(self,channel_generator):
        """generate training channels"""
        i=0
        rho_t = 0
        gamma_t = 0
        varpi = 0.4
        theta_bar_t = torch.zeros(2,self.M)
        recorder_point = [self.Af-1]
        for ix_Af in range(self.Af):
            # print('frame number is ====')
            # print(ix_Af)
            """test delay performance"""
            if ix_Af in recorder_point:
                h_set,gi_set,gj_set = channel_generator.produce_data_set(self.As,self.config)
                dataSet = list(zip(h_set,gi_set,gj_set))
                self.ThetaNN.eval()
                ix_As = 0
                for h,gi,gj in dataSet:
                    htilde = self.ThetaNN(h,gi,gj).detach()
                    self.compute_rate(htilde)
                    self.compute_offloading_ratio_and_delay() 
                    if self.whether_heuristic_matching == False:
                        self.compute_matching_strategy()
                    self.delay_recorder[ix_Af,ix_As] = self.compute_final_delay()
                    ix_As += 1
            # print(self.rate_matrix)
            # print(self.offloading_ratio_matrix)
            # print(self.delay_matrix)
            # print(self.matching_vector)
                
            # print('average delay is =====')
            # print(torch.mean(self.delay_recorder[ix_Af,:]))
            """collect a sample and update phase shifters"""
            self.ThetaNN.train()
            h_set,gi_set,gj_set = channel_generator.produce_data_set(1,self.config)
            dataSet = list(zip(h_set,gi_set,gj_set))
            loss = 0
            for h,gi,gj in dataSet: 
                htilde = self.ThetaNN(h,gi,gj)
                self.compute_rate(htilde)
                self.compute_offloading_ratio_and_delay()
                if self.whether_heuristic_matching == False:
                    self.compute_matching_strategy()
                loss = self.compute_final_delay()
                    
            self.optimizer.zero_grad()
            loss.backward()                
            for name, parms in self.ThetaNN.named_parameters():
                if parms.grad is not None:
                    if name == "theta":                       
                        rho_t = (20/(i+20))**(0.6) 
                        gamma_t = 15/(15+i)
                        theta_bar_t = (1-rho_t)*theta_bar_t + rho_t*parms.grad
                        parms.data = parms.data - gamma_t*theta_bar_t/(2*varpi)
                        # print("grad value is")
                        # print(torch.mean(torch.abs(parms.grad)))
            i=i+1


class randomIRS_performance(MISSCA):
    def __init__(self,config):
        super(randomIRS_performance,self).__init__(config)
        self.M = 0
        self.delay_recorder = torch.zeros(self.As)
        
    def run_MISSCA(self,channel_generator):
        """generate training channels"""

        h_set,gi_set,gj_set = channel_generator.produce_data_set(self.As,self.config)
        dataSet = list(zip(h_set,gi_set,gj_set))
        self.ThetaNN.eval()
        ix_As = 0
        for h,gi,gj in dataSet:
            htilde = self.ThetaNN(h,gi,gj).detach()
            self.compute_rate(htilde)
            self.compute_offloading_ratio_and_delay()                
            self.compute_matching_strategy()
            self.delay_recorder[ix_As] = self.compute_final_delay()
            ix_As += 1
            

class noIRS_performance(MISSCA):
    def __init__(self,config):
        super(noIRS_performance,self).__init__(config)
        self.M = 0
        self.delay_recorder = torch.zeros(self.As)
        
    def run_MISSCA(self,channel_generator):
        """generate training channels"""

        h_set,gi_set,gj_set = channel_generator.produce_data_set(self.As,self.config)
        dataSet = list(zip(h_set,gi_set,gj_set))
        self.ThetaNN.eval()
        ix_As = 0
        for h,gi,gj in dataSet:
            h = h.squeeze(1).squeeze(1)
            self.compute_rate(h)
            self.compute_offloading_ratio_and_delay()                
            self.compute_matching_strategy()
            self.delay_recorder[ix_As] = self.compute_final_delay()
            ix_As += 1


class single_performance(MISSCA):
    def __init__(self,config):
        super(single_performance,self).__init__(config)
        self.loss_recorder = torch.zeros(self.Af)
    
    def find_heuristic_matching_strategy(self):
        Li_index = np.argsort(-self.Li)
        fj_index = np.argsort(-self.fj)
        for i in range(self.I):
            self.matching_vector[Li_index[i]] = fj_index[i]
        
        
    def run_MISSCA(self,channel_generator):
        """generate training channels"""
        i=0
        rho_t = 0
        gamma_t = 0
        varpi = 0.4
        theta_bar_t = torch.zeros(2,self.M)
        h_set,gi_set,gj_set = channel_generator.produce_data_set(1,self.config)
        for ix_Af in range(self.Af):
            """generate a sample and update phase shifters"""
            self.ThetaNN.train()
            
            dataSet = list(zip(h_set,gi_set,gj_set))
            loss = 0
            for h,gi,gj in dataSet: 
                htilde = self.ThetaNN(h,gi,gj)
                self.compute_rate(htilde)
                self.compute_offloading_ratio_and_delay()
                self.compute_matching_strategy()
                loss = self.compute_final_delay()
            
            self.loss_recorder[ix_Af] = loss.detach()
            self.optimizer.zero_grad()
            loss.backward()                
            for name, parms in self.ThetaNN.named_parameters():
                if parms.grad is not None:
                    if name == "theta":                       
                        rho_t = (20/(i+20))**(0.6) 
                        gamma_t = 15/(15+i)
                        theta_bar_t = (1-rho_t)*theta_bar_t + rho_t*parms.grad
                        parms.data = parms.data - gamma_t*theta_bar_t/(2*varpi)
                        # print("grad value is")
                        # print(torch.mean(torch.abs(parms.grad)))
            i=i+1    
        
    
if __name__ == '__main__':
    seed = 20210517
    torch.manual_seed(seed)  
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    test_number = 120
    M_number = 7
    missca_rate = torch.zeros(M_number,test_number)
    randomIRS_rate = torch.zeros(M_number,test_number*3)
    noIRS_rate = torch.zeros(M_number,test_number*3)
    heuristic_rate = torch.zeros(M_number,test_number)
    single_time_rate = torch.zeros(M_number,test_number)
    
    
    config = Config()
    location = LocationSetting(config)
    position = location.getLocation()    
    plt.figure()    
    [position_IRS,position_up_users,position_down_users] = position
    for i in range(config.I):
        plt.scatter(position_up_users[0,i], position_up_users[1,i],c='r')       
    for j in range(config.J):
        plt.scatter(position_down_users[0,j], position_down_users[1,j],c='g')
    
    for ix_M in range(M_number):
        config.M = 2**(ix_M)
        channel_generator =  My_rician_channel(config.I, config.J, 1, 1, config.M, position)
        print('outer iteration number ===============================')
        print(ix_M)
        
        """find a good initial point first"""
        #best_phase = torch.zeros(2,config.M)
        # missca = MISSCA_performance(config)
        # missca.run_MISSCA(channel_generator)
        # min_delay = torch.mean(missca.delay_recorder[-1,:])
        # best_phase = missca.ThetaNN.theta.detach()
        # for ini_ix in range(int(test_number/2)):
        #     missca = MISSCA_performance(config)
        #     missca.run_MISSCA(position)
        #     current_delay = torch.mean(missca.delay_recorder[-1,:])
        #     if current_delay < min_delay:
        #         min_delay = current_delay
        #         best_phase = missca.ThetaNN.theta.detach()
                        
        
        for ix in range(test_number):
            """MISSCA"""
            missca = MISSCA_performance(config)
            #missca.ThetaNN.theta.data = best_phase #initialize
            missca.run_MISSCA(channel_generator)
            missca_rate[ix_M,ix] = torch.mean(missca.delay_recorder[-1,:])           
            
            """heuristic matching"""
            heuristic_ssca = MISSCA_performance(config,True)
            heuristic_ssca.run_MISSCA(channel_generator)
            heuristic_rate[ix_M,ix] = torch.mean(heuristic_ssca.delay_recorder[-1,:])
            
            single_ssca = single_performance(config)
            single_ssca.ThetaNN.theta.data = missca.ThetaNN.theta.detach()
            single_ssca.run_MISSCA(channel_generator)
            single_time_rate[ix_M,ix] = torch.mean(single_ssca.loss_recorder[-3:])
                 
        for ix in range(test_number*3):            
            """random IRS"""
            random_irs = randomIRS_performance(config)
            random_irs.run_MISSCA(channel_generator)
            randomIRS_rate[ix_M,ix] = torch.mean(random_irs.delay_recorder)            
            
            # plt.figure()
            # xAxis = np.linspace(1,config.Af,config.Af)
            # yAxis = torch.mean(missca.delay_recorder,1).detach().numpy()
            # plt.plot(xAxis,yAxis)
            
            """no IRS"""
            noirs = noIRS_performance(config)
            noirs.run_MISSCA(channel_generator)
            noIRS_rate[ix_M,ix] = torch.mean(noirs.delay_recorder)
            


        
    plt.figure()
    x = range(M_number)
    showRange = 20
    temp1 = torch.sort(missca_rate,1)[0][:,:showRange]
    plt.plot(x,torch.mean(temp1,1),'r')
    plt.plot(x,torch.mean(randomIRS_rate,1),'g')
    plt.plot(x,torch.mean(noIRS_rate,1),'b')
    temp2 = torch.sort(heuristic_rate,1)[0][:,:showRange]
    plt.plot(x,torch.mean(temp2,1),'y')
    temp3 = torch.sort(single_time_rate,1)[0][:,:showRange*2]
    plt.plot(x,torch.mean(temp3,1),'c')
    
    plt.legend(['missca','random','no','heuristic','single'])
    plt.show()