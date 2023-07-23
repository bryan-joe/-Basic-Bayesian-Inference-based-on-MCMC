# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 11:43:27 2023

@author: zou qing
@email: zouqing277@gmail.com
"""
import numpy as np
import seaborn as sns
import pyreadr
import pandas as pd
from matplotlib import pyplot as plt


class Gibbs_sampler():
    """In this calss, we define a new object for performing MCMC for posterior inference of 
    the Normal Hierarchical Model.
    """
    def __init__(self, data, group_count, group_mean):
        self.data = data
        self.cc= group_count # stores the number of observations in each group
        self.mm = group_mean  # stores the mean of observations in each group
        self.m = len(data)  # stores the number of groups
        self.n = np.sum(group_count) # stores the total number of observations
        
        # Initialize the hyperparameters
        self.nu_0 = 1.0
        self.sigma2_0 = 100.0
        self.eta_0 = 1.0
        self.tau2_0 = 100.0
        self.mu_0 = 50.0
        self.gamma2_0 = 25.0
        
        # Initialize the Markov chain
        self.mu = 50.0
        self.tau2 = 25.0
        self.sigma2 = 85.0
        self.theta = np.array([40.0] * 100)
               
    def sample_theta(self):
        """This method we need to sample theta vector, a latent variable"""
        for i in range(self.m):
            vtheta = 1.0 / (self.cc[i]/self.sigma2 + 1.0/self.tau2)
            etheta = vtheta * (self.mm[i]*self.cc[i]/self.sigma2 + self.mu/self.tau2)
            self.theta[i] = np.random.normal(etheta, np.sqrt(vtheta)) 
        print(self.theta[:3])    
        
    def sample_sigma2(self):
        nun = self.nu_0 + self.n
        ssn = self.nu_0 * self.sigma2_0
        for i in range(self.m):
            ssn += np.sum((self.data[float(i+1)].values-self.theta[i])**2)
        self.sigma2 =1.0/np.random.gamma(nun*0.5, 2.0/ssn)
        print(self.sigma2)
            
            
    def sample_mu(self):
        vmu = 1.0/(self.m/self.tau2 + 1.0/self.gamma2_0)
        emu = vmu * (self.m * np.mean(self.theta)/self.tau2 + self.mu_0/self.gamma2_0)
        self.mu = np.random.normal(emu, np.sqrt(vmu))
        print(self.mu)
            
    def sample_tau2(self):
        etam = self.eta_0 + self.m
        tss = self.eta_0*self.tau2_0 + np.sum((self.theta-self.mu)**2)
        self.tau2 = 1.0/np.random.gamma(0.5*etam, 2.0/tss)  
        print(self.tau2)
                

result = pyreadr.read_r('nels.RData')
ELS_data = result['Y.school.mathscore']
grouped = ELS_data['mathscore'].groupby(ELS_data['school'])  
data={name:group for name, group in grouped}  
A = grouped.count().values 
B = grouped.mean().values         
Iters = 2000

np.random.seed(2645) 
Theta = np.zeros((100, Iters), dtype=float)
Others = np.zeros((3, Iters), dtype=float)


gibbs_ms = Gibbs_sampler(data, A, B)  
for i in range(Iters):
    print('The '+str(i)+'th iters:')
    print()
    #print(gibbs_ms.m, gibbs_ms.n)
    gibbs_ms.sample_theta()
    gibbs_ms.sample_sigma2()
    gibbs_ms.sample_mu()
    gibbs_ms.sample_tau2()
    Theta[:, i] = gibbs_ms.theta
    Others[0, i] = gibbs_ms.sigma2
    Others[1, i] = gibbs_ms.mu
    Others[2, i] = gibbs_ms.tau2
              
# fig = plt.figure(figsize=(12, 12))
# sns.set_theme(style='ticks')
# ax1 = fig.add_subplot(311)
# ax1.plot(Others[1, 0:1000],"bo--", linewidth=0.7)
# ax2 = fig.add_subplot(312)
# ax2.plot(Others[0, 0:1000],"bo--", linewidth=0.7)  
# ax3 = fig.add_subplot(313)
# ax3.plot(Others[2, 0:1000],"bo--", linewidth=0.7)   
# plt.tight_layout()     
# plt.savefig('SMT_trace.pdf', dpi=300)
# plt.close()        


a = np.mean(Others[1, 0:Iters])
b = np.mean(Others[0, 0:Iters])
c = np.mean(Others[2, 0:Iters]) 
   
fig2 = plt.figure(figsize=(7, 3.5))    
sns.set_theme(style='ticks',font_scale=0.7)    
ax1 = fig2.add_subplot(131)
ax2 = fig2.add_subplot(132)
ax3 = fig2.add_subplot(133)    
kwargs={'linewidth':1.0} 

#plot subplot1   
sns.kdeplot(x=Others[1, 0:Iters], fill=True, color='k' , ax=ax1, **kwargs)
ax1.vlines(x=a, ymin=0, ymax=0.7, colors='r', linestyles='dashed')
ax1.set_xlabel(r'$\mu$')
# subplot2
sns.kdeplot(x=Others[0, 0:Iters], fill=True, color='k' , ax=ax2, **kwargs)
ax2.vlines(x=b, ymin=0, ymax=0.138, colors='r', linestyles='dashed')
ax2.set_xlabel(r'$\sigma^2$')
#subplot3
sns.kdeplot(x=Others[2, 0:Iters], fill=True, color='k' , ax=ax3, **kwargs)
ax3.vlines(x=c, ymin=0, ymax=0.09, colors='r', linestyles='dashed')
ax3.set_xlabel(r'$\tau^2$')
plt.tight_layout()

plt.savefig('Mpdf.pdf', dpi=300)
plt.close()





















