# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12th 2022

Basic model selection paradigm applied to a sophisticated inference scheme

Given observable couples of performances (o,u), can we select the most probable model ?
"""
import sys,os

from cmath import pi
from turtle import width
from imageio import save
import numpy as np
from pyai.base.miscellaneous_toolbox import smooth_1D_array
from pyai.base.matrix_functions import *
from pyai.base.function_toolbox import spm_wnorm
from pyai.neurofeedback_run import save_model_performance_dictionnary,load_model_performance_dictionnary
import matplotlib.pyplot as plt
from pyai.models_neurofeedback.article_1_simulations.climb_stairs_flat_priors import nf_model,nf_model_imp5,nf_model_imp4,evaluate_container
from pyai.neurofeedback_run import trial_plot_from_name
from pyai.model.active_model import ActiveModel
from pyai.model.active_model_save_manager import ActiveSaveManager
from pyai.layer.mdp_layer import mdp_layer

if __name__ == "__main__":
    Nf = 1

    # initial_state
    D_ =[]
    D_.append(np.array([1,1,0,0,0])) #[Terrible state, neutral state , good state, great state, excessive state]
    D_ = normalize(D_)

    d_ =[]
    d_.append(np.zeros(D_[0].shape))

    # State Outcome mapping and beliefs
    # Prior probabilities about initial states in the generative process
    Ns = [5] #(Number of states)
    No = [5]

    # Observations : just the states 
    A_ = []
    # Generally : A[modality] is of shape (Number of outcomes for this modality) x (Number of states for 1st factor) x ... x (Number of states for nth factor)
    pa = 1
    A_obs_mental = np.array([[pa  ,0.5-0.5*pa,0         ,0         ,0   ],
                            [1-pa,pa        ,0.5-0.5*pa,0         ,0   ],
                            [0   ,0.5-0.5*pa,pa        ,0.5-0.5*pa,0   ],
                            [0   ,0         ,0.5-0.5*pa,pa        ,1-pa],
                            [0   ,0         ,0         ,0.5-0.5*pa,pa  ]])
    A_ = [A_obs_mental]



    # prior_ratio = 5 # Correct_weights = ratio*incorrect_weights --> The higher this ratio, the better the quality of the priors
    # prior_strength = 10.0 # Base weight --> The higher this number, the stronger priors are and the longer it takes for experience to "drown" them \in [0,+OO[
    a_ = [np.eye(No[0])]

    # Transition matrixes between hidden states ( = control states)
    pb = 1

    nu = 5
    prop_poublle = 0.3
    npoubelle = int((prop_poublle/(1-prop_poublle))*nu)
    B_ = []
    B_mental_states = np.zeros((Ns[0],Ns[0],nu+npoubelle))

    # Line = where we're going
    # Column = where we're from
    B_mental_states[:,:,0] = np.array([ [1  ,1  ,1,1,1],         # Try to move to terrible state from others
                                        [0  ,0  ,0,0,0],
                                        [0  ,0  ,0,0,0],
                                        [0  ,0  ,0,0,0],
                                        [0  ,0  ,0,0,0]])

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
    for k in range(nu,nu+npoubelle):
        B_mental_states[:,:,k] = normalize(np.ones((5,5)))
        B_mental_states[:,:,k] = np.eye(5)
    B_.append(B_mental_states)


    # We start with a bad action prior
    b_ = [np.ones((B_[0].shape))]


    # Preferences
    la = -2
    rs = 2
    C_mental = np.array([[2*la],
                        [la],
                        [rs],
                        [3*rs],
                        [14*rs]])
    C_ = [C_mental]

    NU = nu + npoubelle


    # Possible actions
    Np = NU #Number of potential actions
    Nf = 1 #Number of state factors

    U_ = np.zeros((NU,Nf)).astype(int)
    U_[:,0] = range(NU)

    #Habits
    E_ = None
    e_ = np.ones((Np,))


    T = 10

    
    savemanager = ActiveSaveManager(T,trial_savepattern=1,intermediate_savepattern=0,verbose=False,modelname="test_fit",folder_name="none")
                                    # Trial related save , timestep related save
    nf_model = ActiveModel(savemanager)
    
    nf_model.T = T
    nf_model.A = A_
    nf_model.a = a_
    nf_model.B = B_
    nf_model.b = b_
    nf_model.C = C_
    nf_model.D = D_
    nf_model.d = d_
    nf_model.U = U_

    nf_model.layer_options.learn_a = False
    nf_model.layer_options.learn_b = True
    nf_model.layer_options.learn_d = False

    nf_model.layer_options.T_horizon = 2
    nf_model.layer_options.learn_during_experience = False
    nf_model.verbose = False

    nf_model.initialize_n_layers(1)

