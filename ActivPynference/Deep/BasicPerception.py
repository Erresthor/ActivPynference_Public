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

T = 100
D = np.array([0.5,0.5])

######## Defining likelihood matrices  ##########

A1 = np.zeros((2,2))
A1[:,0] = [0.75,0.25]
A1[:,1] = [0.25,0.75]
gammaA1 = np.zeros(T)

betaA1m = np.zeros(2)
betaA1m[:] = [0.5,2.0]

X_1_prior = np.zeros((2,T))        ### perception prior (standard vs oddball)
X_1_prior[:,0] = [0.5,0.5]         ### perceptual state prior D1

X_1_posterior = np.zeros((2,T))     ### perception posterior (standard vs oddball)

T = 10

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
beta = 1
precision = np.zeros((T,))
Ni = 1
L = []

for t in range(T):  
    actual_obs= O1bar[:,t]
    
    s_prior = X_1_prior[:,t]
    estimated_obs = np.dot(A1,s_prior)
    
    gammaA1[t] = beta**(-1)
    print(gammaA1[t])
    print(A1)
    normalized_A = softmax_dim2((A1**gammaA1[t]))

    
    
    print(normalized_A)
    print("---------------")
    
    
    
    
#    
#    
#    
#    
#    O1[:,t] = np.dot(A1_prec,X_1_prior[:,t]) #Observation prior
##    print(O1[:,t])
##    print(np.inner(A1_prec,X_1_prior[:,t]))
##    
#    lnA  = np.log(A1)
#    lnA_bar = np.log(A1_prec)
##    
##    
##    betaA1 = betaA1 - nat_log(np.dot(A1,O1bar[:,t]-O1[:,t]))
##    X_1_posterior[:,t] = softmax(nat_log(X_1_prior[:,t])+nat_log(np.dot(A1_prec,O1bar[:,t])))
##    print(X_1_posterior[:,t])
##    print(nat_log(np.sum(np.dot(A1,O1bar[:,t]-O1[:,t]))))
##    
##    AtC = 0
##    for i in range(2):                            ##loop over outcomes
##        for j in range(2):                          ##loop over states
##            
##            
##            AtC += (O1bar[i,t]-A1_prec[i,j])*X_1_posterior[j,t]*np.log(A1[i,j])   ### See "Uncertainty, epistemics and active inference" Parr, Friston.
##    if AtC > betaA1m[0]:
##          AtC = betaA1m[0]-10**-5
##    print(AtC)
##    
###    
###    
####    print(betaA1)
####    print(X_1_posterior) # Observation posterior
####    print('-----------')
###
###    X_1_prior[:,t+1]= 
##
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
