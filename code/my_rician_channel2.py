# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:23:20 2021

@author: yanzhenliu
"""

import numpy as np
from scipy.linalg import sqrtm 
import scipy.special as spl
import math
import torch

def correlation_matrix(M,r):
    Phi = np.zeros([M,M])
    for i in range(M):
        for j in range(M):
            if i <= j:
                Phi[i,j] = r**(j-i)
            else:
                Phi[i,j] = Phi[j,i]
    return Phi

def rician_channel(m,n,Phi1,Phi2, K_factor, h_bar):
    g_random = 2**(-0.5)*(np.random.randn(m,n) + 1j*np.random.randn(m,n))
    g = h_bar*(K_factor/(K_factor+1))**(0.5) + (1/(K_factor+1))**(0.5)*np.matmul(sqrtm(Phi1),np.matmul(g_random,sqrtm(Phi2)))
    return g



class My_rician_channel():
    #K,L,N_t,N_r,N_U,N_D,M_fai = 2,2,16,16,4,4,16
    def __init__(self,K,L,N_U,N_D,M_fai,position,return_los = False):
        self.K = K
        self.L = L
        self.N_U = N_U
        self.N_D = N_D
        self.M_fai = M_fai
        self.return_los = return_los      
        
        
        [position_IRS,position_up_users,position_down_users] = position
        Ny=np.int(np.sqrt(M_fai))
        """compute the distance and angle"""        
        """uplink user--IRS"""
        self.distance_up_IRS = np.zeros(K)
        eve_AOD_up_IRS = np.zeros(K)
        azi_AOD_up_IRS = np.zeros(K)
        self.H_up_IRS_bar = np.zeros([M_fai,N_U,K])+1j*np.zeros([M_fai,N_U,K])
        
        for k in range(K):        
            self.distance_up_IRS[k] = np.linalg.norm(position_up_users[:,k]-position_IRS[:,0])
            eve_AOD_up_IRS[k] = np.arctan((position_IRS[2]-position_up_users[2,k])/np.linalg.norm(position_up_users[0:2,k]-position_IRS[0:2]))
            azi_AOD_up_IRS[k] = np.arctan((position_IRS[0]-position_up_users[0,k])/(position_IRS[1]-position_up_users[1,k]+1e-20))   
            a_t = np.exp(1j*2*np.pi/4*(np.floor((np.arange(M_fai)+1)/Ny)*np.cos(np.pi-eve_AOD_up_IRS[k])+((np.arange(M_fai)+1)-np.floor((np.arange(M_fai)+1)/Ny)*Ny)*np.sin(np.pi-eve_AOD_up_IRS[k])*np.cos(np.pi-azi_AOD_up_IRS[k])))
            a_r = np.exp(1j*2*np.pi/4*(np.arange(N_U))*np.cos(eve_AOD_up_IRS[k]))
            self.H_up_IRS_bar[:,:,k] = np.matmul(np.expand_dims(a_t,1),np.conj(np.expand_dims(a_r,0)))
        
        """downlink user--IRS"""   
        self.distance_IRS_down = np.zeros(L)
        eve_AOD_IRS_down = np.zeros(L)
        azi_AOD_IRS_down = np.zeros(L)
        self.H_IRS_down_bar = np.zeros([N_D,M_fai,L])+1j*np.zeros([N_D,M_fai,L])
        
        for l in range(L):        
            self.distance_IRS_down[l] = np.linalg.norm(position_IRS[:,0]-position_down_users[:,l])
            eve_AOD_IRS_down[l] = np.arctan((position_down_users[2,l]-position_IRS[2])/np.linalg.norm(position_down_users[0:2,l]-position_IRS[0:2]))
            azi_AOD_IRS_down[l] = np.arctan((position_down_users[0,l]-position_IRS[0])/(position_IRS[1]-position_down_users[1,l]+1e-20))
            a_t = np.exp(1j*2*np.pi/4*(np.arange(N_D))*np.cos(np.pi-eve_AOD_IRS_down[l]))
            a_r = np.exp(1j*2*np.pi/4*(np.floor((np.arange(M_fai)+1)/Ny)*np.cos(eve_AOD_IRS_down[l])+((np.arange(M_fai)+1)-np.floor((np.arange(M_fai)+1)/Ny)*Ny)*np.sin(eve_AOD_IRS_down[l])*np.cos(azi_AOD_IRS_down[l])))                               
            self.H_IRS_down_bar[:,:,l] = np.matmul(np.expand_dims(a_t,1),np.conj(np.expand_dims(a_r,0))) 
        
        """uplink--downlink user""" 
        self.distance_up_down = np.zeros([K,L])
        self.H_up_down_bar = np.zeros([N_D,N_U,K,L])+1j*np.zeros([N_D,N_U,K,L])
        for l in range(L):
            for k in range(K):
                self.distance_up_down[k,l] =np.linalg.norm(position_up_users[:,k]-position_down_users[:,l])
                eve_AOD_up_down = np.pi/2
                a_t = np.exp(1j*2*np.pi/4*(np.arange(N_D))*np.cos(np.pi-eve_AOD_up_down))
                a_r = np.exp(1j*2*np.pi/4*(np.arange(N_U))*np.cos(np.pi-eve_AOD_up_down))
                self.H_up_down_bar[:,:,k,l] = np.matmul(np.expand_dims(a_t,1),np.conj(np.expand_dims(a_r,0)))
        
        """generate correlation matrix"""
        r_irs = 0.9
        r_AP = 0.9
        r_user = 0.9
        
        self.Phi_r = correlation_matrix(M_fai,r_irs)
        self.Phi_NU = correlation_matrix(N_U,r_user)
        self.Phi_ND = correlation_matrix(N_D,r_user)      
        
        """rician factor"""
        self.beta_IRS_user = 10**(20/10)
        self.beta_user = 10**(20/10)
        
        """pathloss"""
        self.alpha_IRS_user = 2.2
        self.alpha_user = 3.8
        
        self.C0 = 10**(-3) 
    
    def get_channel_sample(self):
        GU = np.zeros([self.M_fai,self.N_U,self.K]) + 1j*np.zeros([self.M_fai,self.N_U,self.K])  
        GD = np.zeros([self.N_D,self.M_fai,self.L]) + 1j*np.zeros([self.N_D,self.M_fai,self.L])    
        J = np.zeros([self.N_D,self.N_U,self.K,self.L]) + 1j*np.zeros([self.N_D,self.N_U,self.K,self.L])
    
        
        for k in range(self.K):
            GU[:,:,k] = rician_channel(self.M_fai,self.N_U,self.Phi_r,self.Phi_NU,self.beta_IRS_user,self.H_up_IRS_bar[:,:,k])*np.sqrt(self.C0*self.distance_up_IRS[k]**(-self.alpha_IRS_user))
        
        for l in range(self.L):
            GD[:,:,l] = rician_channel(self.N_D,self.M_fai,self.Phi_ND,self.Phi_r,self.beta_IRS_user,self.H_IRS_down_bar[:,:,l])*np.sqrt(self.C0*1e3*self.distance_IRS_down[l]**(-self.alpha_IRS_user))
        
        for l in range(self.L):
            for k in range(self.K):
                J[:,:,k,l] = rician_channel(self.N_D,self.N_U,self.Phi_ND,self.Phi_NU,self.beta_user,self.H_up_down_bar[:,:,k,l])*np.sqrt(self.C0*self.distance_up_down[k,l]**(-self.alpha_user))
                
        # GU_ = np.random.randn(self.M_fai,self.N_U,self.K) + 1j*np.random.randn(self.M_fai,self.N_U,self.K)
        # GD_ = np.random.randn(self.N_D,self.M_fai,self.L) + 1j*np.random.randn(self.N_D,self.M_fai,self.L)    
        # J_ = np.random.randn(self.N_D,self.N_U,self.K,self.L) + 1j*np.random.randn(self.N_D,self.N_U,self.K,self.L)
    
        
        
        
        # for k in range(self.K):
        #     GU_[:,:,k] = H_up_IRS_bar[:,:,k]*np.sqrt(C0*distance_up_IRS[k]**(-alpha_IRS_user))
        
        # for l in range(self.L):
        #     GD_[:,:,l] = H_IRS_down_bar[:,:,l]*np.sqrt(C0*distance_IRS_down[l]**(-alpha_IRS_user))
        
        # for l in range(self.L):
        #     for k in range(self.K):
        #         J_[:,:,k,l] = H_up_down_bar[:,:,k,l]*np.sqrt(C0*distance_up_down[k,l]**(-alpha_user))
                    
                
        # if return_los:    
        #     return J_,GU_,GD_
        # else:
        return J,GU,GD
        
    def produce_data_set(self,num_data,config):
        J_set = []
        GU_set = []
        GD_set = []    
            
        for i in range(num_data):    
            J,GU,GD = self.get_channel_sample()
            """numpy to tensor"""        
            GU = np2tensor(GU).to(dtype=config.dtype,device=config.device)
            GD = np2tensor(GD).to(dtype=config.dtype,device=config.device)
            J = np2tensor(J).to(dtype=config.dtype,device=config.device)
            GU = GU*np.sqrt(config.scale_factor)
            GD = GD*np.sqrt(config.scale_factor)
            J = J*config.scale_factor
    
            J_set.append(J)
            GU_set.append(GU)
            GD_set.append(GD)
            
        return J_set,GU_set,GD_set

def addDelayToChannel(H,tau):
    T = 2.2e-3
    fd=5*10**9*1000/3600/(3*10**8)
    tau_d=math.floor(tau/T)
    
    v = spl.j0(2*np.pi*fd*1*T)
    Rx = spl.j0(2*np.pi*fd*0*T)
    
    a = -v/Rx
    sigma_p = spl.j0(0)
    sigma_p += a*spl.j0(-2*np.pi*fd*1*T)
    
    Hdelay = H
    for i in range(tau_d): 
        Hdelay = -a*Hdelay + math.sqrt(sigma_p)*np.ones_like(H)*np.random.randn(1)
    
    return Hdelay



def np2tensor(A):
    """convert a complex type numpy array to a tensor object"""
    return torch.cat((torch.from_numpy(A.real).unsqueeze(0),torch.from_numpy(A.imag).unsqueeze(0)),0)

def tensor2np(A):
    Are = A[0,:].detach()
    Aim = A[1,:].detach()
    
    return Are.numpy() + 1j*Aim.numpy()
