# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 09:16:10 2021

@author: cjsan
"""
import numpy as np
from spm12_implementation import MDP 
from Deep.MDP_deep import MDP_deep
import random
import sys

def Mindfulness_model():
    print("Mindfulness --- Model set-up ...  ",end='')
    #Points within a trial
    T = 100
    Ni = 16
    
    
    # Priors about initial states
    # Prior probabilities about initial states in the generative process
    D_ =[]
    # Context state factor
    D_.append(np.array([0.5,0.5])) #[Left better, right better]
    # Behaviour state factor
  
    A_ = []
    
    A1 = np.zeros((2,2))
    avari = 0.99
    A1[:,0] = [avari,1-avari]
    A1[:,1] = [1-avari,avari]
    A_.append(A1)
    
    
    bvari = 0.45
    
    B = np.array([[[bvari],[1-bvari]],
              [[1-bvari],[bvari]]])

    B = np.array([[[0.70],[0.6]],
              [[0.3],[0.4]]])
    
    B_ = []
    #a. Transition between context states --> The agent cannot act so there is only one :
    B_.append(B)
    

    C_ = [np.zeros((2,))]
    
    
    # Policies
    Np = 1 #Number of policies
    Nf = 1 #Number of state factors
    V_ = np.zeros((T-1,Np,Nf))
    
    
    #Outcomes
    O = np.zeros((1,T))             ### observations (standard vs oddball)
    O[0,int(T/5)]=1;              ### generative process determined by experimenter
    O[0,int(T/5):int(2*T/5)]=1;              ### generative process determined by experimenter
    O[0,int(3*T/5)]=1;              ### generative process determined by experimenter
    O[0,int(4*T/5)]=1;              ### generative process determined by experimenter
    
    
    
    
    model = MDP_deep()
    
    model.T = T
    model.Ni = Ni
    model.A_ = A_
    
    model.D_ = D_
    
    model.B_  = B_
#    model.b_ = b_
    
   # model.C_ = C_
    
    model.V_ = V_
    model.o = O
    #model.E_ = E_
    #model.e_ = e_
    
    #Other parameters
    model.eta = 1 #Learning rate
    model.beta = 1 # expected precision in EFE(pi), higher beta --> lower expected precision
             # low beta --> high influence of habits, less deterministic policiy selection
    model.alpha = 32 # Inverse temperature / Action precision
                # How much randomness in selecting actions
                # (high alpha --> more deterministic)
    model.erp = 4  # degree of belief resetter at each time point
             # ( 1 means no reset bet. time points, higher values mean more loss in confidence)
    model.tau = 1 #Time constant for evidence accumulation
                # magnitude of updates at each iteration of gradient descent
                # high tau --> smaller updates, smaller convergence, greater  stability
    print("Done")
    return model

m = Mindfulness_model()
m.run()
m.show()