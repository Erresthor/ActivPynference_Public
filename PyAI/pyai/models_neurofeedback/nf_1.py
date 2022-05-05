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
from .neurofeedback_base import NF_model_displayer

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
        a_[0][i,i] = 0.101
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
    No = [A_[0].shape[0]]
    
    
    C_mental = np.array([[la],
                        [0.5*la],
                        [0],
                        [0.5*rs],
                        [rs]])
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
    
    layer.T = 50
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

def custom_run(layer,times,gt_transition,gt_perception,initial_state,initial_observation=None):
    run_comps = []
    for i in range(times):
        next_real_state = initial_state
        next_observation = initial_observation

        layer.prep_trial()
        for t in range(self.T):
            layer.s[:,t] = next_real_state
            if isField(next_observation) :
                layer.o[:,t] = next_observation
            layer.tick()

            # Calculate real states and observations given our own observation rule
            previous_real_states = layer.s[:,t]
            previous_action_chosen = layer.u[:,t]

            next_real_state = gt_transition(previous_real_states,previous_action_chosen)
            next_observation = gt_perception(next_real_state)
        layer.postrun()
        run_comps.append(self.return_run_components())
    return run_comps

def f(previous_state,action_chosen):
    """ temporal state evolution ground trut function for the neurofeedback paradigm. Dictates the evolution of cognitive states depending on the previous state
    and the chosen action. """
    next_state = previous_state
    return next_state

def g(state):
    """ state - observation ground truth correspondance function for the neurofeedback paradigm"""
    mental_phy = phy(state)

    acquired_sig = measure(mental_phy)

    calculated_marker = process(acquired_sig)

    observation = calculated_marker
    return observation

def nf_1_run():
    lay = nf_1_model(5,-2)
    lay.options.learn_during_experience = True
    K = 1500

    B = lay.B_[0]
    b = normalize(lay.b_[0],)
    multi_matrix_plot([B,b], ["Real B","Prior b"])

    A = lay.A_[0]
    a = normalize(lay.a_[0])
    multi_matrix_plot([A,a], ["Real A","Prior a"])

    
    # ticks_indicator = lay.C_[0].shape[0]
    # ticks_mental = lay.D_[0].shape[0]
    # # print(ticks_indicator,ticks_mental)

    # displayer = NF_model_displayer(ticks_mental,ticks_indicator)
    # imL = []
    # for t in range(lay.T):
    #     screen = lay.o[0,t] + 1
    #     mental = lay.s[0,t] + 1
    #     displayer.update_values(screen,mental)
    #     im = displayer.draw((900,900))
    #     print(screen,mental)
    #     print("@")
    #     imL.append(im)
    # path =  'gif_nf_training.gif'
    # imL[0].save(path,save_all=True,append_images=imL[1:],duration=150,loop = 0)

    max_per_line = 8
    i = 1
    j = K
    if (K > max_per_line):
        i = (K//max_per_line)
        j = max_per_line
        if(K%max_per_line!=0):
            i += 1
    fig,axes = plt.subplots(nrows = j,ncols= i)


    evals = []
    for k in range(K):
        print(k)
        if (k==10):
            verbose = True
        else : 
            verbose = False
        lay.verbose = verbose


        lay.run()

        timesteps = np.linspace(0,lay.T-1,lay.T).astype(np.int)
        i = k//max_per_line
        j = k%max_per_line
        if (K>max_per_line):
            axi = axes[j,i]
        else :
            axi = axes[j]
        im = axi.plot(timesteps,lay.s[0,:],label = "Real state")
        im = axi.plot(timesteps[:-1],lay.K[:],label = "Action chosen")
        im = axi.imshow(lay.X[0], interpolation = 'nearest',origin='lower')
        axi.title.set_text("exp"+"_"+str(k))

        #print(evaluate_run(lay.efficacy, lay.o))
        evals.append(evaluate_run(lay.efficacy, lay.o)[0])
    #fig.show()
    # lay.run()

    path = "D:\\data\\"+ "test\\" + "evals_1.txt"
    save_flexible(evals, path)



    B = lay.B_[0]
    b = normalize(lay.b_[0],)
    multi_matrix_plot([B,b], ["Real B","Learnt B"])

    A = lay.A_[0]
    a = normalize(lay.a_[0])
    multi_matrix_plot([A,a], ["Real A","Learnt A"])

    fig,axes = plt.subplots()
    axes.plot(range(K),evals)
    fig.show()
    input()
