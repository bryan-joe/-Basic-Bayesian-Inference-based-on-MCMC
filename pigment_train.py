# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:07:41 2023

@author: LENOVO
"""
import pandas as pd
import numpy as np
from gibbs_pigment import gibbs_method_first
from gibbs_pigment import gibbs_method_second
from matplotlib import pyplot as plt


np.random.seed(14578)
df = pd.read_table('pigment.dat', sep=' ', encoding='utf-8')
data=np.zeros((15,2))
data2=np.asarray(df['Moisture'])

for i in range(15):
    for j in range(2):
        data[i, j]=data2[2*i+j] 

var_alpha=86
var_beta=58
var_epsilon=1
burn_in=1000
iter = 10000
mu_list=[]
gibbs_pig_11 = gibbs_method_first(data, var_alpha, var_beta, var_epsilon)
for i in range(burn_in):
    print('\n')
    print('Iter: '+str(i))
    gibbs_pig_11.update_mu()
    gibbs_pig_11.update_alpha()
    gibbs_pig_11.update_beta()
    
for i in range(iter):
    print('\n')
    print('Iter: '+str(i))
    gibbs_pig_11.update_mu()
    gibbs_pig_11.update_alpha()
    gibbs_pig_11.update_beta()
    print(gibbs_pig_11.mu)
    mu_list.append(gibbs_pig_11.mu)


fig1=plt.figure(figsize=(9,6))
plt.plot(mu_list, color='b', linewidth=1)
plt.title('$\mu$ sample path')
plt.xlabel('iters')
plt.savefig('mu_pigment.png', dpi=200)
plt.close(fig1)


mu_list=[]
gibbs_pig_22 = gibbs_method_second(data, var_alpha, var_beta, var_epsilon)
for i in range(burn_in):
    print('\n')
    print('Iter: '+str(i))
    gibbs_pig_22.update_mu()
    gibbs_pig_22.update_gamma()
    gibbs_pig_22.update_eta()
    
for i in range(iter):
    print('\n')
    print('Iter: '+str(i))
    gibbs_pig_22.update_mu()
    gibbs_pig_22.update_gamma()
    gibbs_pig_22.update_eta()
    print(gibbs_pig_22.mu)
    mu_list.append(gibbs_pig_22.mu)


fig2=plt.figure(figsize=(9,6))
plt.plot(mu_list, color='r', linewidth=1)
plt.title('$\mu$ sample path of method 2')
plt.xlabel('iters')
plt.savefig('mu2_pigment.png', dpi=200)
plt.close(fig2)    