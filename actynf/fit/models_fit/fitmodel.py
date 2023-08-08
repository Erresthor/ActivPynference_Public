# -*- coding: utf-8 -*-
"""
Created on Tue Aug 3 10:55:21 2021

@author: cjsan
"""
import numpy as np
import random
from PIL import Image, ImageDraw
import sys,os,inspect
import math
import matplotlib.pyplot as plt

from ...base.miscellaneous_toolbox import flexible_copy , isField
from ...base.function_toolbox import normalize
from ...base.function_toolbox import spm_dot,spm_kron
from ...base.plotting_toolbox import multi_matrix_plot
from ...base.file_toolbox import load_flexible,save_flexible
from ...layer_old.mdp_layer import mdp_layer
from ...layer_old.layer_postrun import evaluate_run

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
     
# SET UP MODEL STRUCTURE ------------------------------------------------------
#print("-------------------------------------------------------------")
#print("------------------SETTING UP MODEL STRUCTURE-----------------")
#print("-------------------------------------------------------------")

def fitmodel(T):
    initial_state = 0

    la = -2
    rs = 2

    print("FIT --- Model set-up ...  ")
    #Points within a trial
    
    Ni = 16

    # Priors about initial states
    # Prior probabilities about initial states in the generative process
    D_ =[]
    # Context state factor
    D_.append(np.array([0,0,0,0,0])) #[Terrible state, neutral state , good state, great state, excessive state]
    D_[0][initial_state] = 1


    # Prior beliefs about initial states in the generative process
    d_ =[]
    # Mental states
    d_.append(np.array([0.25,0.5,0.2,0.025,0.025])) #[Terrible state, neutral state , good state, great state, excessive state]
    
    
    # State Outcome mapping and beliefs
    # Prior probabilities about initial states in the generative process
    Ns = [5] #(Number of states)
    No = [5]

    # Observations : just the states 
    A_ = []
    #Mapping from states to observed hints, accross behaviour states (non represented)
    #
    # [ .  . ]  No hint
    # [ .  . ]  Machine Left Hint            Rows = observations
    # [ .  . ]  Machine Right Hint
    # Left Right
    # Columns = context state

    # Generally : A[modality] is of shape (Number of outcomes for this modality) x (Number of states for 1st factor) x ... x (Number of states for nth factor)
    A_obs_mental = np.zeros((No[0],Ns[0]))
    # When attentive, the feedback is modelled as perfect :
    A_obs_mental[:,:] = np.array([[1,0,0,0,0],
                                    [0,1,0,0,0],
                                    [0,0,1,0,0],
                                    [0,0,0,1,0],
                                    [0,0,0,0,1]])

    # A_obs_mental[:,:,1] = np.random.random((5,5))
    A_ = [A_obs_mental]
    a_ = []
    for mod in range (len(A_)):
        a_.append(np.copy(A_[mod]))
    a_[0] = np.ones((a_[0].shape))*0.1
    for i in range(5):
        a_[0][i,i] = 10

    
    # Transition matrixes between hidden states ( = control states)
    B_ = []
    #a. Transition between context states --> The agent cannot act so there is only one :
    nu = 3
    B_mental_states = np.zeros((Ns[0],Ns[0],nu))

    
    # Line = where we're going
    # Column = where we're from
    pb = 1


    B_mental_states[:,:,0] = np.array([ [1  ,1  ,0,0,0],         # Try to move to terrible state from others
                                        [0  ,0  ,1,0,0],
                                        [0  ,0  ,0,1,0],
                                        [0  ,0  ,0,0,1],
                                        [0  ,0  ,0,0,0]])

    # B_mental_states[:,:,0] = np.array([ [1  ,0  ,0,0,0],         # Try to move to terrible state from others
    #                                     [0,1,0,0,0],
    #                                     [0  ,0  ,1,0,0],
    #                                     [0  ,0  ,0,1,0],
    #                                     [0  ,0  ,0,0,1]])

    
    B_mental_states[:,:,1] = np.array([[1,0  ,0  ,0  ,0  ],         # Try to move to neutral state from others
                                        [0 ,1  ,0  ,0  ,0  ],
                                        [0  ,0  ,1  ,0  ,0  ],
                                        [0  ,0  ,0  ,1  ,0  ],
                                        [0  ,0  ,0  ,0  ,1 ]])
    
    B_mental_states[:,:,2] = np.array([ [0  ,0   ,0  ,0   ,0  ],         # Try to move to good state from others
                                        [1  ,0   ,0  ,0   ,0  ],
                                        [0  ,1   ,0  ,0   ,0  ],
                                        [0  ,0   ,1  ,0   ,0  ],
                                        [0  ,0   ,0  ,1   ,1  ]])
    B_.append(B_mental_states)


    b_ = []
    for fac in range (len(B_)):
        b_.append(np.copy(B_[fac])*100)
    b_[0] = np.ones((b_[0].shape))*0.1
    
    b_ = B_
    # b_[0][1,:,:] = 0.15
    # b_[0][2,:,:] = 0.2
    # b_[0][3,:,:] = 0.3
    # b_[0][4,:,:] = 0.4

    # for i in range(B_[0].shape[-1]):
    #     b_[0][:,:,i][B_[0][:,:,i]>0.5] += 1
    
    # b_[0][:,:,:] += 3*np.random.random((b_[0].shape))
    # To encourage exploration, we expect rather positive outcomes for all actions


    # Preferred outcomes
    # One matrix per outcome modality. Each row is an observation, and each
    # columns is a time point. Negative values indicate lower preference,
    # positive values indicate a high preference. Stronger preferences promote
    # risky choices and reduced information-seeking.
    
    
    C_mental = np.array([[la],
                        [0.5*la],
                        [0],
                        [0.5*rs],
                        [rs]])
    C_ = [C_mental]
    
    
    # Policies
    Np = 5 #Number of policies
    Nf = 2 #Number of state factors


    U_ = np.array([[0],
                   [1],
                   [2]])
    
    #Habits
    E_ = None
    e_ = np.ones((Np,))
    
    layer = mdp_layer()
    
    layer.T = T
    layer.options.T_horizon = 2
    layer.Ni = Ni
    layer.A_ = A_
    layer.a_ = a_
    
    layer.D_ = D_
    layer.d_ = d_
    
    layer.B_  = B_
    layer.b_ = b_

    layer.C_ = C_
    
    layer.U_ = U_
    
    #model.E_ = E_
    #model.e_ = e_
    
    #Other parameters
    return layer