# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:08:23 2023

@author: LENOVO
"""

"""
In this sub-file, we need to define a class for Gibbs sampling algorithm
"""
import numpy as np
from numpy import random


class gibbs_method_first():
    def __init__(self, y, var_alpha, var_beta, var_epsilon):
        self.y=np.asarray(y)
        self.I=y.shape[0]
        self.J=y.shape[1]
        self.n=y.size
        #self.hyperparams=hyperparams
        self.var_alpha=var_alpha
        self.var_beta=var_beta
        self.var_epsilon=var_epsilon
        
        #Initialize some latent variables
        self.mu = float(0)
        # alpha vector is a one-dimensional array sith size I
        self.alpha = random.normal(loc=[0] * self.I, scale=[np.sqrt(self.var_alpha)]*self.I) 
        # beta array is a two-dimensional array of size (self.I, self.J)
        self.beta = random.normal(loc=np.zeros((self.I, self.J), dtype=int), scale=np.full((self.I, self.J),np.sqrt(self.var_beta), dtype=int))
    
    
    def update_mu(self):
        """This time sample mu, grand average"""
        
        posterior_mean=np.mean(self.y)-self.J*np.sum(self.alpha)/self.n-np.mean(self.beta)
        posterior_std=np.sqrt(self.var_epsilon/self.n)
        self.mu = random.normal(loc=posterior_mean, scale=posterior_std)
        
    def update_alpha(self):
        """because alpha are independent of each other, so we can adopt block sampling"""
        var=self.var_epsilon*self.var_alpha/(self.J*self.var_alpha+self.var_epsilon)
        post_std=np.full(self.I, np.sqrt(var), dtype=float)
        post_mean=1/self.var_epsilon*var*self.J*(np.mean(self.y, 1)- np.full(self.I, self.mu)-np.mean(self.beta, 1))
        self.alpha=random.normal(loc=post_mean, scale=post_std)
        
    def update_beta(self):
        """similarly, we can block sample beta"""
        
        var=self.var_beta*self.var_epsilon/(self.var_beta+self.var_epsilon)
        A = np.full(self.y.shape, self.mu)
        B = np.expand_dims(self.alpha, 1).repeat(2, axis=1)
        postmean_array=var/self.var_epsilon*(self.y-A-B)
        poststd_array=np.full(self.beta.shape, np.sqrt(var))
        self.beta=random.normal(loc=postmean_array, scale=poststd_array)


class gibbs_method_second(): 
    def __init__(self, y, var_alpha, var_beta, var_epsilon):
        self.y=np.asarray(y)
        self.I=y.shape[0]
        self.J=y.shape[1]
        self.n=y.size
        #self.hyperparams=hyperparams
        self.var_alpha=var_alpha
        self.var_beta=var_beta
        self.var_epsilon=var_epsilon
        
        # Initializing
        self.mu = float(0)
        #gamma vector is of dimension 15*1
        self.gamma = random.normal(loc=[0]*self.I, scale=[self.var_alpha]*self.I)
        #eta array is of size 15*2
        aaa = np.expand_dims(self.gamma,1).repeat(self.J, axis=1)
        self.eta = random.normal(loc=aaa, scale=np.full(self.y.shape, self.var_beta))
        
    def update_mu(self):
        postmean=np.mean(self.gamma)
        post_std=np.sqrt(self.var_alpha/self.I)
        self.mu=random.normal(loc=postmean, scale=post_std)
        
    def update_gamma(self):
        var = 1/(self.J/self.var_beta + 1/self.var_alpha)
        post_std = np.full(self.I, np.sqrt(var))
        post_mean = var*(np.sum(self.eta, 1)/self.var_beta+np.full(self.I, self.mu)/self.var_alpha)
        self.gamma = random.normal(loc=post_mean, scale=post_std)
        
    def update_eta(self):
        var=self.var_beta*self.var_epsilon/(self.var_beta+self.var_epsilon)
        B=np.expand_dims(self.gamma, 1).repeat(2, axis=1)
        postmean_array=var*(self.y/self.var_epsilon+B/self.var_beta)
        poststd_array=np.full(self.eta.shape, np.sqrt(var))
        self.eta = random.normal(loc=postmean_array, scale=poststd_array)
        
    