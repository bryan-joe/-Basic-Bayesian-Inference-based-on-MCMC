# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 19:08:18 2023

@author: LENOVO
"""

import numpy as np
import pandas as pd
from numpy import random
from scipy.stats import norm
from matplotlib import pyplot as plt

class IndependentChain_MH():
    """This method adopts Beta(1,1) as a proposal"""
    def __init__(self, data, prior_a, prior_b):
        self.y=data
        self.n=data.size
        self.a = prior_a
        self.b = prior_b
        
        # Initialize x chain
        self.x = random.beta(self.a, self.b)
        self.xx = []
        
        
    def calculate_likelihood(self, delta):
        h=0
        for i in range(self.n):
            h += np.log(norm.pdf(self.y[i], loc=7, scale=0.5)*delta+(1-delta)*norm.pdf(self.y[i], loc=10, scale=0.5))
        return np.exp(h)
        
    def sample_delta(self):
        x_candidate = random.beta(self.a, self.b)
        u = random.uniform()
        
        acceptance_ratio = self.calculate_likelihood(x_candidate)/self.calculate_likelihood(self.x)
        if u < acceptance_ratio:
            self.x = x_candidate
            self.xx.append(self.x)
        else:
            self.xx.append(self.x)

            
df = pd.read_table('mixture.dat', encoding='utf-8')
data = np.asarray(df['y'])

random.seed(12765)  
burn_in = 50
iters = 1000
prior_a = 1
prior_b = 1

gibbs11 = IndependentChain_MH(data, prior_a, prior_b)
array1 = []
for i in range(burn_in):
    #print('\n')
    print('Iters: '+str(i))
    gibbs11.sample_delta()
  
     
for i in range(iters):
    print('Iters: '+str(i))
    gibbs11.sample_delta()
    print(gibbs11.x)
    array1.append(gibbs11.x)
     
random.seed(12765)
prior_a2 = 2
prior_b2 = 10    
gibbs22 = IndependentChain_MH(data, prior_a2, prior_b2)
array2 = []
 
for i in range(burn_in):
    #print('\n')
    print('Iters: '+str(i))
    gibbs22.sample_delta()
  
     
for i in range(iters):
    print('Iters: '+str(i))
    gibbs22.sample_delta()
    print(gibbs22.x)
    array2.append(gibbs22.x)
     
    
fig=plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(array1, color='b', linewidth=1,label='beta(1,1)')
ax1.legend()
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(array2, color='b', linewidth=1,label='beta(2,10)') 
ax2.legend()

plt.savefig('IC_MH_comparison.png', dpi=200)
plt.close(fig)
   
    
 
    
 