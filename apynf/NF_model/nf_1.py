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


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from base.spm12_implementation import MDP 
from base.miscellaneous_toolbox import flexible_copy 
from base.function_toolbox import normalize
from mdp_layer import mdp_layer
from neurofeedback_base import NF_model_displayer
from base.plotting_toolbox import multi_matrix_plot
import matplotlib.pyplot as plt

from base.function_toolbox import spm_dot,spm_kron
     
# SET UP MODEL STRUCTURE ------------------------------------------------------
#print("-------------------------------------------------------------")
#print("------------------SETTING UP MODEL STRUCTURE-----------------")
#print("-------------------------------------------------------------")
#

class nf_model:
    def __init__(self):
        displayer = NF_model_displayer()


def nf_1_model(rs,la):
    initial_state = 0



    print("Explore-Exploit --- Model set-up ...  ",end='')
    #Points within a trial
    T = 500
    Ni = 16
    
    Nf = 2

    # Priors about initial states
    # Prior probabilities about initial states in the generative process
    D_ =[]
    # Context state factor
    D_.append(np.array([0,0,0,0,0])) #[Terrible state, neutral state , good state, great state, excessive state]
    D_[0][initial_state] = 1

    # Behaviour state factor
    D_.append(np.array([0.0,1])) #{'attentive','distracted'}
    
    # Prior beliefs about initial states in the generative process
    d_ =[]
    # Mental states
    d_.append(np.array([0.25,0.5,0.2,0.025,0.025])) #[Terrible state, neutral state , good state, great state, excessive state]
    # Behaviour beliefs
    d_.append(np.array([0.5,0.5]))  #{'attentive','distracted'}
    
    
    # State Outcome mapping and beliefs
    # Prior probabilities about initial states in the generative process
    Ns = [D_[0].shape[0],D_[1].shape[0]] #(Number of states)
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
    A_obs_mental = np.zeros((No[0],Ns[0],Ns[1]))
    # When attentive, the feedback is modelled as perfect :
    A_obs_mental[:,:,0] = np.array([[1,0,0,0,0],
                                    [0,1,0,0,0],
                                    [0,0,1,0,0],
                                    [0,0,0,1,0],
                                    [0,0,0,0,1]])
    # When distracted, the feedback is modelled as noisy :
    pa = 1
    A_obs_mental[:,:,1] = np.array([[pa  ,0.5-0.5*pa,0         ,0         ,0   ],
                                    [1-pa,pa        ,0.5-0.5*pa,0         ,0   ],
                                    [0   ,0.5-0.5*pa,pa        ,0.5-0.5*pa,0   ],
                                    [0   ,0         ,0.5-0.5*pa,pa        ,1-pa],
                                    [0   ,0         ,0         ,0.5-0.5*pa,pa  ]])

    # A_obs_mental[:,:,1] = np.random.random((5,5))
    A_ = [A_obs_mental]
    a_ = []
    for mod in range (len(A_)):
        a_.append(np.copy(A_[mod]))
    a_[0] = np.ones((a_[0].shape))*0.1
    for i in range(5):
        a_[0][i,i] = 5
    # Transition matrixes between hidden states ( = control states)
    B_ = []
    #a. Transition between context states --> The agent cannot act so there is only one :
    nu = 5
    B_mental_states = np.zeros((Ns[0],Ns[0],nu))

    
    # Line = where we're going
    # Column = where we're from
    pb = 1


    B_mental_states[:,:,0] = np.array([ [1  ,1  ,1,1,1],         # Try to move to terrible state from others
                                        [0  ,0  ,0,0,0],
                                        [0  ,0  ,0,0,0],
                                        [0  ,0  ,0,0,0],
                                        [0  ,0  ,0,0,0]])

    # B_mental_states[:,:,0] = np.array([ [1  ,0  ,0,0,0],         # Try to move to terrible state from others
    #                                     [0,1,0,0,0],
    #                                     [0  ,0  ,1,0,0],
    #                                     [0  ,0  ,0,1,0],
    #                                     [0  ,0  ,0,0,1]])

    
    B_mental_states[:,:,1] = np.array([[1-pb,0  ,0  ,0  ,0  ],         # Try to move to neutral state from others
                                        [pb ,1  ,1  ,1  ,1  ],
                                        [0  ,0  ,0  ,0  ,0  ],
                                        [0  ,0  ,0  ,0  ,0  ],
                                        [0  ,0  ,0  ,0  ,0 ]])
    
    B_mental_states[:,:,2] = np.array([ [1  ,0   ,0  ,0   ,0  ],         # Try to move to good state from others
                                        [0  ,1-pb,0  ,0   ,0  ],
                                        [0  ,pb  ,1  ,1   ,1  ],
                                        [0  ,0   ,0  ,0   ,0  ],
                                        [0  ,0   ,0  ,0   ,0  ]])
    
    B_mental_states[:,:,3] = np.array([ [1  ,0  ,0   ,0  ,0  ],         # Try to move to target state from others
                                        [0  ,1  ,0   ,0  ,0  ],
                                        [0  ,0  ,1-pb,0  ,0  ],
                                        [0  ,0  ,pb  ,1  ,1  ],
                                        [0  ,0  ,0   ,0  ,0  ]])
    
    B_mental_states[:,:,4] = np.array([ [1  ,0  ,0  ,0  ,1-pb],         # Try to move to best state from others
                                        [0  ,1  ,0  ,0  ,0  ],
                                        [0  ,0  ,1  ,0  ,0  ],
                                        [0  ,0  ,0  ,1-pb,0  ],
                                        [0  ,0  ,0  ,pb ,pb]])
    B_.append(B_mental_states)


    p = 0.98    
    p = 1
    B_attention_state = np.array([[p,1-p],
                                  [1-p,p]])
    B_.append(np.expand_dims(B_attention_state,-1))
    b_ = []
    for fac in range (len(B_)):
        b_.append(np.copy(B_[fac])*100)
    b_[0] = np.ones((b_[0].shape))*0.1
    b_[0][1,:,:] = 0.15
    b_[0][2,:,:] = 0.2
    b_[0][3,:,:] = 0.3
    b_[0][4,:,:] = 0.4
    # To encourage exploration, we expect rather positive outcomes for all actions


    # Preferred outcomes
    # One matrix per outcome modality. Each row is an observation, and each
    # columns is a time point. Negative values indicate lower preference,
    # positive values indicate a high preference. Stronger preferences promote
    # risky choices and reduced information-seeking.
    No = [A_[0].shape[0]]
    
    
    C_mental = np.array([[la],
                        [0],
                        [rs/2],
                        [rs],
                        [5*rs]])
    C_ = [C_mental]
    
    
    # Policies
    Np = 5 #Number of policies
    Nf = 2 #Number of state factors


    U_ = np.zeros((nu,Nf)).astype(np.int)
    U_[:,0] = range(nu)
    
    #Habits
    E_ = None
    e_ = np.ones((Np,))
    
    
    
    
    
    
    layer = mdp_layer()
    
    layer.T = 100
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
    print("Done")
    return layer

if (__name__ == "__main__"):
    lay = nf_1_model(20,-2)
    # print(lay.A_)
    # print(lay.A_[0][:,:,1])
    # print((normalize(lay.A_))[:,:,1])
    lay.run()
    #print(lay.trees[0].to_string(3))
    print("-----------------------------------------------------------------------------")
    # print(lay.s)
    # print(lay.K)
    # print(lay.X[0].shape)
    # print(lay.u)
    timesteps = np.linspace(0,lay.T-1,lay.T).astype(np.int)
    plt.plot(timesteps,lay.s[0,:],label = "Real state")
    plt.plot(timesteps[:-1],lay.K[:],label = "Action chosen")
    plt.imshow(lay.X[0], interpolation = 'nearest',origin='lower')
    plt.legend()
    plt.show()

    B = lay.B_[0]
    b = normalize(lay.b_[0],)
    multi_matrix_plot([B,b], ["Real B","Perceived B"])


    A = lay.A_[0]
    a = normalize(lay.a_[0])
    multi_matrix_plot([A,a], ["Real A","Perceived A"])

    ticks_indicator = lay.C_[0].shape[0]
    ticks_mental = lay.D_[0].shape[0]
    # print(ticks_indicator,ticks_mental)

    displayer = NF_model_displayer(ticks_mental,ticks_indicator)
    imL = []
    for t in range(lay.T):
        screen = lay.o[0,t] + 1
        mental = lay.s[0,t] + 1
        displayer.update_values(screen,mental)
        im = displayer.draw((900,900))
        print(screen,mental)
        print("@")
        imL.append(im)
    path =  'gif_nf_training.gif'
    imL[0].save(path,save_all=True,append_images=imL[1:],duration=150,loop = 0)
    
    timesteps = np.linspace(0,lay.T-1,lay.T).astype(np.int)
    plt.plot(timesteps,lay.s[0,:],label = "Real state")
    plt.plot(timesteps[:-1],lay.K[:],label = "Action chosen")
    plt.imshow(lay.X[0], interpolation = 'nearest',origin='lower')
    plt.legend()
    plt.show()

    B = lay.B_[0]
    b = normalize(lay.b_[0],)
    multi_matrix_plot([B,b], ["Real B","Perceived B"])


    A = lay.A_[0]
    a = normalize(lay.a_[0])
    multi_matrix_plot([A,a], ["Real A","Perceived A"])


    lay.run()


    timesteps = np.linspace(0,lay.T-1,lay.T).astype(np.int)
    plt.plot(timesteps,lay.s[0,:],label = "Real state")
    plt.plot(timesteps[:-1],lay.K[:],label = "Action chosen")
    plt.imshow(lay.X[0], interpolation = 'nearest',origin='lower')
    plt.legend()
    plt.show()

    B = lay.B_[0]
    b = normalize(lay.b_[0],)
    multi_matrix_plot([B,b], ["Real B","Perceived B"])


    A = lay.A_[0]
    a = normalize(lay.a_[0])
    multi_matrix_plot([A,a], ["Real A","Perceived A"])



    # # # fig,ax = plt.subplots()
    



    # # initial_state = 1
    # # D_ =[]
    # # # Context state factor
    # # D_.append(np.array([0,0,0,0,0])) #[Terrible state, neutral state , good state, great state, excessive state]
    # # D_[0][initial_state] = 1
    # # # Behaviour state factor
    # # D_.append(np.array([1,0])) #{'attentive','distracted'}

    # # d = spm_kron(D_)

    # # b_kron = [] 
    # # for k in range(5) :
    # #     b_kron.append(1)
    # #     for f in range(2):
    # #         b_kron[k] = spm_kron(b_kron[k],lay.B_[f][:,:,lay.U_[k,f]])  
    # # print(spm_kron(lay.B_))

    # # for act in range(5):
    # #     print("#")
    # #     print(act)
    # #     print(np.dot(b_kron[act],d))

