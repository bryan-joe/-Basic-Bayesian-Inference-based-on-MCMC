# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 20:11:56 2023

@author: zou qing
@email: zouqing277@gmail.com
"""
import pyreadr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import invwishart
from scipy.stats import poisson



def log_mvn(beta, theta, Sigma):
    iSigma = np.linalg.inv(Sigma)
    C = np.matmul((beta-theta).T, iSigma)
    s = np.matmul(C, beta-theta)
    return -0.5*s

def log_pos(beta, X_aug, y):
    """
    Here y is any vector of count data in each group, beta is a regression coefficients
    """
    mu = np.exp(np.matmul(X_aug, beta))    
    c = poisson.logpmf(y, mu)
    return np.sum(c)

class Gibbs():
    """This algorithm is intended to perform hybrid MCMC for 
        Hierarchical posisson regression mixed effects model"""
    def __init__(self, Y, X_aug, p, Beta):
        self.Y = Y
        self.m = Y.shape[0]
        self.X_aug = np.asarray(X_aug) # design matrix 20*5 in each group
        self.p = p
        
        # prior distribution hyperparameters setting
        self.mu0 = np.mean(Beta, axis=0)
        self.S0 = np.cov(Beta.T)
        self.eta0 = self.p + 2
        self.iL0 = np.linalg.inv(self.S0)
        self.iSigma = self.iL0
        
        # initial value
        self.theta = np.mean(Beta, axis=0)
        self.Sigma = np.cov(Beta.T)
        self.Beta = Beta # a matrix of size 21 * 5
        
    # sample lambda
    def sample_theta(self):
        #sample grand mean vector theta
        Lm = np.linalg.inv(self.iL0 + self.m*self.iSigma)
        C = np.matmul(self.iL0, self.mu0) + np.matmul(self.iSigma, np.sum(self.Beta, axis=0))
        mum = np.matmul(Lm, C)
        self.theta = np.random.multivariate_normal(mean=mum, cov=Lm)
    
    # update Sigma
    def sample_Sigma(self):
        D = self.Beta - self.theta
        Stheta = np.matmul(D.T, D)
        S = np.linalg.inv(Stheta+self.S0)
        self.Sigma = invwishart.rvs(df=self.eta0+self.m, scale=S)
        
    # update beta
    def sample_Beta(self):
        # Note that in this model we use the metropolis algorithm, instead of MH algorithm
        for j in range(self.m):
            beta_star = np.random.multivariate_normal(self.Beta[j, :], 0.5*self.Sigma)
            aa = log_pos(beta_star, self.X_aug, self.Y.loc[j]) + log_mvn(beta_star, self.theta, self.Sigma)
            bb = log_pos(self.Beta[j,:].T, self.X_aug, self.Y.loc[j])+log_mvn(self.Beta[j, :].T, self.theta, self.Sigma)
            log_ratio =  aa-bb
            s = np.random.uniform()
            if np.log(s) < log_ratio:
                self.Beta[j, :] = beta_star[:]
 
  
 
                        
    
result = pyreadr.read_r('tumorLocation.RData')
Y = result['tumorLocation']
Iters = 50000
p = 5
xs = np.linspace(0.05,1.00,20)
X_aug = np.zeros((20, 5))
for i in range(5):
    X_aug[:, i] = xs ** (i)
m = Y.shape[0]
Ymean = Y.mean(axis='index')

 
# Empirical Bayesain Prior
Beta = np.zeros((m, p))
for i in range(m):
    reg4 = LinearRegression().fit(X_aug[:,1:5], np.log(Y.loc[i]+0.05))
    Beta[i, 0] = reg4.intercept_
    Beta[i, 1:p] = reg4.coef_


# perform MCMC 50000 times
gibbs_sa1 = Gibbs(Y, X_aug, p, Beta) 
theta_array = np.zeros((p, Iters))
Beta_array = np.zeros((m, p, Iters))
for i in range(Iters):
    print(' ')
    print('The '+str(i)+'th iteration:')
    gibbs_sa1.sample_theta()
    gibbs_sa1.sample_Sigma()
    gibbs_sa1.sample_Beta()
    theta_array[:, i] = gibbs_sa1.theta
    Beta_array[:, :, i] = gibbs_sa1.Beta













