# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 21:32:42 2023

@author: zou qing
@email: zouqing277@gmail.com
"""
import pyreadr
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns




class Gibbs():
    """in this class, we define gibbs sampler for the normal hierarchical model"""
    def __init__(self, y1=[], y2=[]):
        self.y1 = np.asarray(y1)
        self.y2 = np.asarray(y2)
        self.n1 = len(y1)
        self.n2 = len(y2)
        
        # Initialize the hyperparameters. Indeed, these parts can be incorporated into argument
        self.mu_0 = 50.0
        self.gamma_0 = 625.0
        self.delta_0 = 0.0
        self.tau_0 = 625.0
        self.nu_0 = 1.0
        self.sigma_0 = 100.0
        #Initial the zero state of mu, delta, sigma_square
        self.mu = 0.5*(np.mean(self.y1) + np.mean(self.y2))
        self.delta = 0.5*(np.mean(self.y1) - np.mean(self.y2))
        self.sigma_square = 1.0
        
        self.y1_average = np.mean(self.y1)
        self.y2_average = np.mean(self.y2)
        
    def sample_mu(self):
        gamma_n = 1/(1/self.gamma_0 + (self.n1 + self.n2)/self.sigma_square)
        mu_n = gamma_n * (self.mu_0/self.gamma_0 +(self.n1*(self.y1_average-self.delta)
                    +self.n2*(self.y2_average+self.delta))/self.sigma_square )
        self.mu = np.random.normal(mu_n, np.sqrt(gamma_n))
        
    def sample_delta(self):
        tau_n = 1/(1/self.tau_0 + (self.n1+self.n2)/self.sigma_square)
        delta_n = tau_n * (self.delta_0/self.tau_0 +(self.n1*(self.y1_average-self.mu)-self.n2*(self.y2_average-self.mu))/self.sigma_square )
        self.delta = np.random.normal(tau_n, np.sqrt(delta_n))
        
    def sample_sigma_square(self):
        nu_n = self.nu_0 + self.n1 + self.n2
        sigma_n = self.nu_0*self.sigma_0 + np.sum((self.y1-self.mu-self.delta)**2)+np.sum((self.y2-self.mu+self.delta)**2)
        temp_sigma = np.random.gamma(nu_n*0.5, 2.0/sigma_n)   
        self.sigma_square = 1.0 / temp_sigma
        
        
result = pyreadr.read_r('nels.RData')        
df2 = result['y.school1']
df3 = result['y.school2']  
y1 = df2['y.school1'].values
y2 = df3['y.school2'].values 

gibbs_sampler = Gibbs(y1, y2)         
            
np.random.seed(1)
list_mu = []
list_delta = []
list_sigma_square = []
for i in range(5000):
    print('The '+str(i)+'th iteration:') 
    gibbs_sampler.sample_sigma_square()
    gibbs_sampler.sample_mu()
    gibbs_sampler.sample_delta()
    list_sigma_square.append(gibbs_sampler.sigma_square)
    list_mu.append(gibbs_sampler.mu)
    list_delta.append(gibbs_sampler.delta)        
                    
fig = plt.figure()
sns.set_theme(style='ticks', font_scale=1)
ax1 = fig.add_subplot(121)
sns.kdeplot(x=list_mu, color='k',fill=True, ax=ax1)
 
ax1.set_xlabel(r'$\mu$')

ax2 = fig.add_subplot(122)
sns.kdeplot(x=list_delta, color='k',fill=True, ax=ax2)
ax2.vlines(x=0.0, ymin=0,  ymax=0.25, linestyles='dashed')
ax2.set_xlabel(r'$\delta$')
fig.suptitle(r"The marginal posterior distribution of $\mu$ and $\delta$")

plt.tight_layout()
plt.savefig('two_school.pdf', dpi=300)
plt.close()

        
        
        
        
        
        
        
    
    
    
    