# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:44:16 2021

@author: Administrator
"""
import numpy as np

class LocationSetting(object):
    def __init__(self,config):
        self.I = config.I
        self.J = config.J
        self.R1 = config.R1
        self.R2 = config.R2
        self.O1 = config.O1
        self.O2 = config.O2
        self.heighti = config.heighti
        self.heightu = config.heightu
    
    def getLocation(self):
        I_position = np.random.rand(3,self.I)-0.5
        J_position = np.random.rand(3,self.J)-0.5
        IRS_position = np.expand_dims(np.array([0,0,self.heighti]),1)
        
        I_position[2,:] = self.heightu
        J_position[2,:] = self.heightu
        
        for i in range(self.I):           
            radius = self.R1*np.sqrt(np.random.rand(1))
            theta = 2*np.pi*np.random.rand(1)
            I_position[0,i] = radius*np.cos(theta)+self.O1
            I_position[1,i] = radius*np.sin(theta)


        for j in range(self.J):
            radius = self.R2*np.sqrt(np.random.rand(1))
            theta = 2*np.pi*np.random.rand(1)
            J_position[0,j] = radius*np.cos(theta)+self.O2
            J_position[1,j] = radius*np.sin(theta)
        
        return [IRS_position,I_position,J_position]
        
