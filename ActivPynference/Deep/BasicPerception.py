# -*- coding: utf-8 -*-
"""
Created on Wed May 26 09:42:55 2021

@author: cjsan
"""

import numpy as np
import sys
sys.path.append('..')
from function_toolbox import *
from plotting_toolbox import *
import matplotlib.pyplot as plt
from miscellaneous_toolbox import clamp

def softmax(X):                                                                 ###converts log probabilities to probabilities
  norm = np.sum(np.exp(X)+10**-5)
  Y = (np.exp(X)+10**-5)/norm
  return Y

def softmax_dim2(X):                                                            ###converts matrix of log probabilities to matrix of probabilities
  norm = np.sum(np.exp(X)+10**-5,axis=0)
  Y = (np.exp(X)+10**-5)/norm
  return Y

def normalise(X):                                                               ###normalises a matrix of probabilities
  X= X/np.sum(X,0)
  return X




####### Mind-wandering during an odd-ball perceptual task 
####### Two-level model with single factor at each level

################################################################################
### Defining parameters
################################################################################

T = 500
D = np.array([0.5,0.5])

######## Defining likelihood matrices  ##########

A1 = np.zeros((2,2))
A1[:,0] = [0.75,0.25]
A1[:,1] = [0.25,0.75]
gammaA1 = np.zeros(T)

betaA1m = np.zeros(2)
betaA1m[:] = [0.5,2.0]

beta_1_prior = np.zeros(T)

X_1_prior = np.zeros((2,T))        ### perception prior (standard vs oddball)
X_1_prior[:,0] = [0.5,0.5]         ### perceptual state prior D1

X_1_posterior = np.zeros((2,T))     ### perception posterior (standard vs oddball)

O = np.zeros(T)             ### observations (standard vs oddball)
O[int(T/5)]=1;              ### generative process determined by experimenter
O[int(T/5):int(2*T/5)]=1;              ### generative process determined by experimenter
O[int(3*T/5)]=1;              ### generative process determined by experimenter
O[int(4*T/5)]=1;              ### generative process determined by experimenter

O1 = np.zeros((2,T))        ### observation prior (standard vs oddball)
O1bar = np.zeros((2,T))     ### observation posterior (standard vs oddball)
for t in range(T):
  O1bar[int(O[t]),t]=1



betaA1 = 1
## Basic
#for t in range(T-1):
#    s_prior = X_1_prior[:,t]
#    
#    observation_prior = np.dot(A1,s_prior)
#    
#    observation_posterior = O1bar[:,t]
#    
#    state_likelihood_given_outcomes = np.dot(A1,observation_posterior)
#    
#    
#    print(observation_prior,observation_posterior)
#    print(s_prior,state_likelihood_given_outcomes)
#    
#    state_posterior = softmax(nat_log(s_prior) + nat_log(state_likelihood_given_outcomes))
#    X_1_posterior[:,t] = state_posterior
#    X_1_prior[:,t+1] = state_posterior
#    
#    print('---')
#basic_autoplot(X_1_posterior[0,:-1])


# Basic with precision calculation
def static_precision():
    beta_1_prior[0] = 0.1
    precision = np.zeros((T,))
    Ni = 1
    L = []
    
    for t in range(T):  
        s_prior = X_1_prior[:,t]
        beta_prior = beta_1_prior[t]
        
        gammaA1[t] = beta_prior**(-1)
        normalized_A1 = softmax_dim2((A1**gammaA1[t]))
        
        o_prior = np.dot(normalized_A1,s_prior)
        o_posterior= O1bar[:,t]
        
        #Likelihood of hidden states -->
        # Given the observations, what are the probabilities of having this as a hidden state
        L.append(np.dot(normalized_A1,o_posterior))
        
        s_posterior = softmax(np.log(s_prior)+np.log(L[t]))
        
        
        # for each combination of hidden states :
        # compute (o_posterior - normalized_A1[:,combination])*posterior_belief_about_combination*log(A1[:,combination])
        # e.g. :
        AtC = 0
        for i in range(2) :
            inter = (o_posterior-normalized_A1[:,i])*s_posterior[i]*np.log(A1[:,i])
            AtC += np.sum(inter)
        if(AtC > beta_prior):
            AtC = beta_prior - 1e-5
        beta_posterior = beta_prior - AtC
    
        
        if(t<T-1):
            X_1_prior[:,t+1] = s_posterior
            beta_1_prior[t+1] = beta_posterior
            
        
    l = np.linspace(0,T,T)
    plt.plot(l,X_1_prior[1,:])
    plt.plot(l,beta_1_prior[:])
    


def dynamic_precision():
    beta_1_prior[0] = 1
    precision = np.zeros((T,))
    Ni = 1
    L = []
    
    B = np.array([[[0.8],[0.2]],
                  [[0.2],[0.8]]])
    
    #B = np.array([[[0.7],[0.3]],
    #              [[0.3],[0.7]]])
    #
    #tester = 0.999
    #
    #B = np.array([[[tester],[1-tester]],
    #              [[1-tester],[tester]]])
        
    for t in range(T):  
        print('-----')
        
        s_prior = X_1_prior[:,t]
        beta_prior = beta_1_prior[t]
        
        beta_prior = 1
        
        gammaA1[t] = beta_prior**(-1)
        
        
        limits = 4.5
        gammaA1[t] = clamp(gammaA1[t],-limits,limits) #prevent computational anomalies during normalization
        
        normalized_A1 = softmax_dim2(A1**gammaA1[t])
        A1bar = (A1**gammaA1[t])
        o_prior = np.dot(normalized_A1,s_prior)
        
        
        o_posterior= O1bar[:,t]
        
        #Likelihood of hidden states -->
        # Given the observations, what are the probabilities of having this as a hidden state
        L.append(np.dot(A1bar,o_posterior))
        s_posterior = softmax(np.log(s_prior)+np.log(L[t]))
        
        # for each combination of hidden states :
        # compute (o_posterior - normalized_A1[:,combination])*posterior_belief_about_combination*log(A1[:,combination])
        # e.g. :
        AtC = 0
        for i in range(2) :
            inter = (o_posterior-normalized_A1[:,i])*s_posterior[i]*np.log(A1[:,i])
            AtC += np.sum(inter)
        if(AtC > 0.5):
            AtC = 0.5
        beta_posterior = beta_prior - AtC
    
    #    if (beta_posterior < 0):
    #        beta_posterior = 0
    #    
        if(t<T-1):
            X_1_prior[:,t+1] = np.inner(B[:,:,0],s_posterior)
            beta_1_prior[t+1] = beta_posterior
        print(s_posterior)
    
    l = np.linspace(1,T,T)
    plt.plot(l,O)
    plt.plot(l,X_1_prior[0,:])
    plt.plot(l,beta_1_prior)




beta_1_prior[0] = 1
precision = np.zeros((T,))
Ni = 1
L = []
B = np.array([[[0.8],[0.2]],
              [[0.2],[0.8]]])
    
for t in range(T):  
    print('-----')
    
    s_prior = X_1_prior[:,t]
    beta_prior = beta_1_prior[t]
    
    beta_prior = 1
    
    gammaA1[t] = beta_prior**(-1)
    
    
    limits = 4.5
    gammaA1[t] = clamp(gammaA1[t],-limits,limits) #prevent computational anomalies during normalization
    
    normalized_A1 = softmax_dim2(A1**gammaA1[t])
    A1bar = (A1**gammaA1[t])
    o_prior = np.dot(normalized_A1,s_prior)
    
    
    o_posterior= O1bar[:,t]
    
    #Likelihood of hidden states -->
    # Given the observations, what are the probabilities of having this as a hidden state
    L.append(np.dot(A1bar,o_posterior))
    s_posterior = softmax(np.log(s_prior)+np.log(L[t]))
    
    # for each combination of hidden states :
    # compute (o_posterior - normalized_A1[:,combination])*posterior_belief_about_combination*log(A1[:,combination])
    # e.g. :
    AtC = 0
    for i in range(2) :
        inter = (o_posterior-normalized_A1[:,i])*s_posterior[i]*np.log(A1[:,i])
        AtC += np.sum(inter)
    if(AtC > 0.5):
        AtC = 0.5
    beta_posterior = beta_prior - AtC

#    if (beta_posterior < 0):
#        beta_posterior = 0
#    
    if(t<T-1):
        X_1_prior[:,t+1] = np.inner(B[:,:,0],s_posterior)
        beta_1_prior[t+1] = beta_posterior
    print(s_posterior)

l = np.linspace(1,T,T)
plt.plot(l,O)
plt.plot(l,X_1_prior[0,:])
plt.plot(l,beta_1_prior)











