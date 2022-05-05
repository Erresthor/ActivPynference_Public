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

from ..base.miscellaneous_toolbox import flexible_copy , isField
from ..base.function_toolbox import normalize
from ..base.function_toolbox import spm_dot,spm_kron
from ..base.plotting_toolbox import multi_matrix_plot
from ..base.file_toolbox import load_flexible,save_flexible
from ..layer.mdp_layer import mdp_layer
from ..layer.layer_postrun import evaluate_run

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


def multistate_model(T):
    print("MULTISTATE --- Model set-up ...  ",end='')

    #Points within a trial
    rs = 2
    la = -2
    initial_state = 0
    Ni = 16
    
    Nf = 2

    # Priors about initial states
    # Prior probabilities about initial states in the generative process
    D_ =[]
    # FACTOR 1 : Context state factor
    D_.append(np.array([0,0,0,0,0])) #[Terrible state, neutral state , good state, great state, excessive state]
    D_[0][initial_state] = 1

    # FACTOR 2 : Behaviour state factor
    D_.append(np.array([1,0])) #{'attentive','distracted'}
    
    # Prior beliefs about initial states in the generative process
    d_ =[]
    # Mental states
    d_.append(np.array([0.25,0.5,0.2,0.025,0.025])) #[Terrible state, neutral state , good state, great state, excessive state]
    # Behaviour beliefs
    d_.append(np.array([0.5,0.5]))  #{'attentive','distracted'}
    
    
    # State Outcome mapping and beliefs
    # Prior probabilities about initial states in the generative process
    Ns = [D_[0].shape[0],D_[1].shape[0]] #(Number of states)
    No = [5,3]



    #---------------------------------------------------------------------------------------------------------
    #--------------------------------               A                -----------------------------------------
    #---------------------------------------------------------------------------------------------------------
    # Observations : just the states 
    A_ = []
    #Mapping from states to observed hints, accross behaviour states (non represented)
    # Generally : A[modality] is of shape (Number of outcomes for this modality) x (Number of states for 1st factor) x ... x (Number of states for nth factor)
    A_obs_mental = np.zeros((No[0],Ns[0],Ns[1]))
    # When attentive, the feedback is modelled as perfect :
    A_obs_mental[:,:,0] = np.array([[1,0,0,0,0],
                                    [0,1,0,0,0],
                                    [0,0,1,0,0],
                                    [0,0,0,1,0],
                                    [0,0,0,0,1]])
    # When distracted, the feedback is modelled as noisy :
    pa = 10 # Noise level
    A_obs_mental[:,:,1] = np.array([[pa  ,0.5-0.5*pa,0         ,0         ,0   ],
                                    [1-pa,pa        ,0.5-0.5*pa,0         ,0   ],
                                    [0   ,0.5-0.5*pa,pa        ,0.5-0.5*pa,0   ],
                                    [0   ,0         ,0.5-0.5*pa,pa        ,1-pa],
                                    [0   ,0         ,0         ,0.5-0.5*pa,pa  ]])
    A_obs_mental[:,:,1] = normalize(A_obs_mental[:,:,0] + pa*np.random.random(A_obs_mental[:,:,1].shape))
    
    A_obs_mental = normalize(A_obs_mental)

    # print(A_obs_mental[:,:,1])

    A_att_perception = np.zeros((No[1],Ns[0],Ns[1]))
    # When attentive, the attentive level is observed with probability pa1 :
    pa1 = 0.5
    A_att_perception[:,:,0] = np.array([[pa1,pa1,pa1,pa1,pa1],    #ATTENTIVE
                                        [1-pa1,1-pa1,1-pa1,1-pa1,1-pa1],    #UNKNOWN
                                        [0,0,0,0,0]])   #DISTRACTED
    
    # When distracted, the distracted level is observed with probability pa2 :
    pa2 = 0.5
    A_att_perception[:,:,1] = np.array([[0,0,0,0,0],                    #ATTENTIVE
                                        [1-pa2,1-pa2,1-pa2,1-pa2,1-pa2],     #UNKNOWN
                                        [pa2,pa2,pa2,pa2,pa2]])              #DISTRACTED
    # A_obs_mental[:,:,1] = np.random.random((5,5))
    A_ = [A_obs_mental,A_att_perception]



    a_ = []
    for mod in range (len(A_)):
        a_.append(np.copy(A_[mod]))
        a_[mod] = np.ones((a_[mod].shape))*0.1
        a_[mod] = 10*A_[mod] + a_[mod]
    


    #---------------------------------------------------------------------------------------------------------
    #--------------------------------               B                -----------------------------------------
    #---------------------------------------------------------------------------------------------------------
    # Transition matrixes between hidden states ( = control states)
    B_ = []
    #a. Transition between context states --> The agent cannot act so there is only one :
    nu = 5
    B_mental_states = np.zeros((Ns[0],Ns[0],nu))
    # Line = where we're going
    # Column = where we're from
    pb = 1
    B_mental_states[:,:,0] = np.array([ [1  ,1  ,0,0,0],         # Try to lower target level
                                        [0  ,0  ,1,0,0],
                                        [0  ,0  ,0,1,0],
                                        [0  ,0  ,0,0,1],
                                        [0  ,0  ,0,0,0]])
    
    B_mental_states[:,:,1] = np.array([[1,0  ,0  ,0  ,0  ],         #|
                                        [0 ,1  ,0  ,0  ,0  ],       #|
                                        [0  ,0  ,1  ,0  ,0  ],      #|
                                        [0  ,0  ,0  ,1  ,0  ],      #|
                                        [0  ,0  ,0  ,0  ,1 ]])      #|
    B_mental_states[:,:,3] = np.array([[1,0  ,0  ,0  ,0  ],         #|------->  Try to stay neutral (x 3)
                                    [0 ,1  ,0  ,0  ,0  ],           #|
                                    [0  ,0  ,1  ,0  ,0  ],          #|
                                    [0  ,0  ,0  ,1  ,0  ],          #|
                                    [0  ,0  ,0  ,0  ,1 ]])          #|
    B_mental_states[:,:,4] = np.array([[1,0  ,0  ,0  ,0  ],         #|
                                    [0 ,1  ,0  ,0  ,0  ],           #|
                                    [0  ,0  ,1  ,0  ,0  ],          #|
                                    [0  ,0  ,0  ,1  ,0  ],          #|
                                    [0  ,0  ,0  ,0  ,1 ]])          #|
    B_mental_states[:,:,2] = np.array([ [1  ,0   ,0  ,0   ,1  ],         # Try to increase target level
                                        [1  ,0,0  ,0   ,0  ],
                                        [0  ,1  ,0  ,0   ,0  ],
                                        [0  ,0   ,1  ,0   ,0  ],
                                        [0  ,0   ,0  ,1   ,0  ]])
    B_.append(B_mental_states)
    nu_att = 2
    B_attentional_action = np.zeros((Ns[1],Ns[1],nu_att))
    pb = 0.99
    B_attentional_action[:,:,0] = np.array([[pb  ,0],
                                            [1-pb,1]])    # Keep attentional state steady
    
    p_refocus = 0.6
    B_attentional_action[:,:,1] = np.array([[1 ,p_refocus    ],
                                            [0 ,1 - p_refocus]])     # Attempt to refocus attentional state
    B_.append(B_attentional_action)

    b_ = []
    for fac in range (len(B_)):
        b_.append(np.copy(B_[fac])*100)
        b_[fac] = np.ones((b_[fac].shape))*0.1
        b_[fac] = b_[fac] + 0.1*B_[fac]

    # Preferred outcomes
    # One matrix per outcome modality. Each row is an observation, and each
    # columns is a time point. Negative values indicate lower preference,
    # positive values indicate a high preference. Stronger preferences promote
    # risky choices and reduced information-seeking.
    No = [A_[0].shape[0]]
    
    
    C_mental = np.array([[la],
                        [0.5*la],
                        [0],
                        [0.5*rs],
                        [rs]])
    C_att = np.array([[0],
                     [0],
                     [0]])
    C_ = [C_mental,C_att]
    
    
    # Policies
    Np = 5 #Number of policies
    Nf = 2 #Number of state factors

    number_of_distinct_policies = nu*nu_att
    U_ = np.zeros((number_of_distinct_policies,Nf)).astype(np.int)   
    
    for k2 in range(nu_att):
        for k1 in range(nu):
            U_[k2*nu+k1,:] = np.array([k1,k2])


    U_ = np.array([[0,0],
                   [0,1],
                   [1,0],
                   [2,0],
                   [3,0],
                   [4,0]])
                
    print(U_)


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
    
    #model.e_ = e_
    
    #Other parameters
    print("Done")
    return layer




def multistate_run():
    lay = multistate_model(50)
    lay.options.learn_during_experience = False
    for k in range(50):
        lay.run()
    

    for fac in range(len(lay.b_)):
        print()
        for u in range(lay.b_[fac].shape[2]) :
            print(u)
            print(np.round(lay.b_[fac][:,:,u]))
    print("----------")

    input()

    for k in range(10):
        lay.run()
        print("-----------------------------------------")
        print("O : ")
        print(lay.O)
        print("States : ")
        print(lay.s)
        print("Ensuing Actions ")
        print(lay.u)
        print("-----------------------------------------")
        print()
        input()