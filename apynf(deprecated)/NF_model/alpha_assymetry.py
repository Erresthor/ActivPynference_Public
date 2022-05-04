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
import random as r
import os.path

from scipy import stats
from mayavi import mlab
import plotly.graph_objects as go

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from base.spm12_implementation import MDP 
from base.miscellaneous_toolbox import flexible_copy , isField , index_to_dist, dist_to_index
from base.function_toolbox import normalize
from mdp_layer import mdp_layer
from neurofeedback_base import NF_model_displayer
from base.plotting_toolbox import multi_matrix_plot,multi_3dmatrix_plot,matrix_plot
from base.file_toolbox import load_flexible,save_flexible
import matplotlib.pyplot as plt

from base.function_toolbox import spm_dot,spm_kron

from layer_postrun import evaluate_run
from layer_learn import MemoryDecayType
from pynf_functions import *

from layer_sumup import *


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
def nf_2_model(rs,la):

    initial_state_1 = 2
    initial_state_2 = 2

    # Priors about initial states
    # Prior probabilities about initial states in the generative process
    D_ =[]
    # Context state factor
    D_.append(np.array([0,0,0,0,0])) # Attentional state
    D_[0][initial_state_1] = 1
    D_.append(np.array([0,0,0,0,0])) # Attentional state
    D_[1][initial_state_2] = 1

    # Prior beliefs about initial states in the generative process
    d_ =[]
    # Mental states
    d_.append(np.array([0.2,0.2,0.2,0.2,0.2])) # Alpha right belief
    d_.append(np.array([0.2,0.2,0.2,0.2,0.2])) # Alpha left belief

    Nf = 2
    Ns = [5,5] #(Number of states)
    No = [5]

    # Observations : corresponding indicator
    A_ = []
    
    # ALPHA ASSYMETRY MARKERS
    # Generally : A[modality] is of shape (Number of outcomes for this modality) x (Number of states for 1st factor) x ... x (Number of states for nth factor)
    A_0 = np.zeros((No[0],Ns[0],Ns[1]))
    pa = 1

    # Ideal observation ensures 1-1 correspondance between observation and hidden state (here the attentionnal level)
    A_0[0,:,:] = np.array([[1   ,0   ,0   ,0   ,0   ],
                           [0   ,1   ,0   ,0   ,0   ],
                           [0   ,0   ,1   ,0   ,0   ],
                           [0   ,0   ,0   ,1   ,0   ],
                           [0   ,0   ,0   ,0   ,1   ]])

    A_0[1,:,:] = np.array([[0   ,1   ,0   ,0   ,0   ],
                           [1   ,0   ,1   ,0   ,0   ],
                           [0   ,1   ,0   ,1   ,0   ],
                           [0   ,0   ,1   ,0   ,1   ],
                           [0   ,0   ,0   ,1   ,0   ]])

    A_0[2,:,:] = np.array([[0   ,0   ,1   ,0   ,0   ],
                           [0   ,0   ,0   ,1   ,0   ],
                           [1   ,0   ,0   ,0   ,1   ],
                           [0   ,1   ,0   ,0   ,0   ],
                           [0   ,0   ,1   ,0   ,0   ]])

    A_0[3,:,:] = np.array([[0   ,0   ,0   ,1   ,0   ],
                           [0   ,0   ,0   ,0   ,1   ],
                           [0   ,0   ,0   ,0   ,0   ],
                           [1   ,0   ,0   ,0   ,0   ],
                           [0   ,1   ,0   ,0   ,0   ]])

    A_0[4,:,:] = np.array([[0   ,0   ,0   ,0   ,1   ],
                           [0   ,0   ,0   ,0   ,0   ],
                           [0   ,0   ,0   ,0   ,0   ],
                           [0   ,0   ,0   ,0   ,0   ],
                           [1   ,0   ,0   ,0   ,0   ]])
    A_ = [A_0]


    a_ = []
    for mod in range (len(A_)):
        a_.append(np.copy(A_[mod]))
    a_[0] = np.ones((a_[0].shape))*0.1


    # Transition matrixes between hidden states ( = control states)
    B_ = []
    #a. Transition between context states --> The agent cannot act so there is only one :
    nu = 5

    B_0 = np.zeros((Ns[0],Ns[0],nu))
    B_left[:,:,1] = np.array([[0  ,0  ,0  ,0  ,0  ],         # Try to move to higher activity
                              [1  ,0  ,0  ,0  ,0  ],
                              [0  ,1  ,0  ,0  ,0  ],
                              [0  ,0  ,1  ,0  ,0  ],
                              [0  ,0  ,0  ,1  ,1 ]])
    
    B_left[:,:,2] = np.array([[0  ,0  ,0  ,0  ,0  ],         # Move to higher activity (fast)
                              [0  ,0  ,0  ,0  ,0  ],
                              [1  ,0  ,0  ,0  ,0  ],
                              [0  ,1  ,0  ,0  ,0  ],
                              [0  ,0  ,1  ,1  ,1 ]])
    
    B_left[:,:,3] = np.array([[1  ,1  ,0  ,0  ,0  ],         # Move to lower activity
                              [0  ,0  ,1  ,0  ,0  ],
                              [0  ,0  ,0  ,1  ,0  ],
                              [0  ,0  ,0  ,0  ,1  ],
                              [0  ,0  ,0  ,0  ,0 ]])
    
    B_left[:,:,4] = np.array([[1  ,1  ,1  ,0  ,0  ],         # Move to lower activity (fast)
                              [0  ,0  ,0  ,1  ,0  ],
                              [0  ,0  ,0  ,0  ,1  ],
                              [0  ,0  ,0  ,0  ,0  ],
                              [0  ,0  ,0  ,0  ,0 ]])

    

    B_right[:,:,0] = np.array([ [1,0,0,0,0],         # Stay at the same activity
                                [0,1,0,0,0],
                                [0,0,1,0,0],
                                [0,0,0,1,0],
                                [0,0,0,0,1]])

    B_right[:,:,1] = np.array([[0  ,0  ,0  ,0  ,0  ],         # Try to move to highrt activity
                               [1  ,0  ,0  ,0  ,0  ],
                               [0  ,1  ,0  ,0  ,0  ],
                               [0  ,0  ,1  ,0  ,0  ],
                               [0  ,0  ,0  ,1  ,1  ]])
    
    B_right[:,:,2] = np.array([[0  ,0  ,0  ,0  ,0  ],         # Move to higher activity (fast)
                               [0  ,0  ,0  ,0  ,0  ],
                               [1  ,0  ,0  ,0  ,0  ],
                               [0  ,1  ,0  ,0  ,0  ],
                               [0  ,0  ,1  ,1  ,1  ]])
    
    B_right[:,:,3] = np.array([[1  ,1  ,0  ,0  ,0  ],         # Move to lower activity
                               [0  ,0  ,1  ,0  ,0  ],
                               [0  ,0  ,0  ,1  ,0  ],
                               [0  ,0  ,0  ,0  ,1  ],
                               [0  ,0  ,0  ,0  ,0  ]])
    
    B_right[:,:,4] = np.array([[1  ,1  ,1  ,0  ,0  ],         # Move to lower activity (fast)
                               [0  ,0  ,0  ,1  ,0  ],
                               [0  ,0  ,0  ,0  ,1  ],
                               [0  ,0  ,0  ,0  ,0  ],
                               [0  ,0  ,0  ,0  ,0  ]])
    B_.append(B_left,B_right)

    b_ = []
    for fac in range (len(B_)):
        b_.append(np.copy(B_[fac])*100)
    b_[0] = np.ones((b_[0].shape))*1

        
    C_mental = np.array([[la    ],
                         [0.5*la],
                         [0     ],
                         [0.5*rs],
                         [rs    ]])
    C_ = [C_mental]
    
    
    # Policies
    nu = 9
    Nf = 2 #Number of state factors


    U_ = np.zeros((nu,Nf)).astype(np.int)
    U_ = np.array([[0,0],
                   [1,0],
                   [0,1],
                   [2,0],
                   [0,2],
                   [3,0],
                   [0,3],
                   [1,3],
                   [3,1]])
    
    #Habits
    E_ = None
    e_ = np.ones((Np,))
    
    
    layer = mdp_layer()
    
    layer.T = 10

    layer.options.T_horizon = 2
    layer.options.learn_during_experience = True
    layer.options.memory_decay = MemoryDecayType.STATIC #MemoryDecayType.NO_MEMORY_DECAY #MemoryDecayType.STATIC
    layer.options.decay_half_time = 1000


    layer.Ni = 16
    layer.A_ = A_
    layer.a_ = a_
    
    layer.D_ = D_
    layer.d_ = d_
    
    layer.B_  = B_
    layer.b_ = b_

    layer.C_ = C_
    
    layer.U_ = U_

    return layer

def nf_generator(layer,gt_transition,gt_perception,initial_state,initial_observation=None):
    next_real_state = initial_state
    next_observation = initial_observation

    layer.prep_trial()
    for t in range(layer.T):
        #compressed_states = dist_to_index(next_real_state)
        #layer.s[:,t] = compressed_states
        layer.s[:,t] = next_real_state

        if isField(next_observation) :
            compressed_observations = dist_to_index(next_observation)
            layer.o[:,t] = compressed_observations

        
        layer.tick()
        yield layer

        # Calculate real states and observations given our own observation rule
        # previous_real_states = index_to_dist(layer.s[:,t],initial_state)
        previous_real_states = layer.s[:,t]
        if (t<layer.T-1):
            previous_action_chosen = layer.u[:,t]

        next_real_state = gt_transition(previous_real_states,previous_action_chosen,layer)
        next_observation = gt_perception(layer.return_void_state_space(populate=next_real_state),layer)
        #print(next_real_state,next_observation)
    layer.postrun()

def custom_run(datapath,K,
            total_savepoints = 10,
            anyplot=False,plotting_trials = False,max_per_line = 8,
            verbose=True, noiselevel = 0,sham = False):
    lay = nf_2_model(5,-2)
    lay.prep_trial()
    lay.options.learn_during_experience = False #false




    if(anyplot):
        B = lay.B_[0]
        b = normalize(lay.b_[0],)
        multi_matrix_plot([B,b], ["Real B","Prior b"])

        A = lay.A_[0]
        a = normalize(lay.a_[0])
        multi_matrix_plot([A,a], ["Real A","Prior a"])

        if (plotting_trials):
            i = 1
            j = K
            if (K > max_per_line):
                i = (K//max_per_line)
                j = max_per_line
                if(K%max_per_line!=0):
                    i += 1
            fig,axes = plt.subplots(nrows = j,ncols= i)


    def f(previous_state,action_chosen,layer):
        """ temporal state evolution ground trut function for the neurofeedback paradigm. Dictates the evolution of cognitive states depending on the previous state
        and the chosen action. """
        returner = []
        for i in range(len(layer.B_)):
            B_mental_states = layer.B_[i]

            posterior_dist = B_mental_states[:,previous_state,action_chosen]

            next_state_dist = np.argwhere(r.random() <= np.cumsum(posterior_dist,axis=0))[0][0]
            
            returner.append(next_state_dist)

        return np.array(returner)

    def g(state_dist,layer):
        """ state - observation ground truth correspondance function for the neurofeedback paradigm --> The BCI PIPELINE"""
        
        def phy(state):  # Return the physiological signal map related to the mental state succession
            mental_phy = state

            def meas_func(X):
                dist = np.linalg.norm(X)
                if (dist < 1):
                    rule = normalize(np.linspace(state_dist.shape[0]-1, 0,state_dist.shape[0]))
                    #print(state,rule/np.max(rule),np.inner(state,rule/np.max(rule))) 
                    return (1-dist)*np.inner(state,rule/np.max(rule))
                    # Rule describes how state intensity translates to physiological activity independently from spatial considerations
                else : 
                    return 0

            return meas_func
        

        mental_phy = phy(state_dist)

        noise_intensity = noiselevel
        def measure(mental_phy): # Return the measure of the physiological signal map.
            
            measure = mental_phy(np.array([0,0,0])) # here we measure at the epicenter of the signal generated

            

            #acquired_sig = 0.5 + np.random.normal(0,1,1)

            if (sham):
                acquired_sig = np.random.random()
            else :
                acquired_sig = measure + noise_intensity*np.random.normal(0,1,1)
            return acquired_sig
        acquired_sig = measure(mental_phy)

        def process(acquired_sig):
            # Let rule be the decoding rule applied. If rule is the same as physiolgical GT, we should try to get the same input

            sef = max(min(1 - acquired_sig,1),0)   # Clamp signal to an acceptable range

            state_estimation_int =  int(np.round((state_dist.shape[0]-1)*sef))

            observation = np.zeros(state_dist.shape)
            observation[state_estimation_int] = 1

            calculated_marker = observation
            # observation_marker = acquired_sig
            return calculated_marker
        calculated_marker = process(acquired_sig)

        observation = calculated_marker
        return observation
    
    # for i in range(30):
    #     sd = lay.return_void_state_space(populate=np.array([2]))
    #     q = g(sd,lay)
    #     print(sd,q)

    # # visualize function :
    # X, Y, Z = np.mgrid[-2:2:40j, -2:2:40j, -2:2:40j]
    # values = np.zeros(X.shape)
    # for kx in range(X.shape[0]):
    #     for ky in range(X.shape[1]):
    #         for kz in range(X.shape[2]):
    #             values[kx,ky,kz] = func(np.array([X[kx,ky,kz],Y[kx,ky,kz],Z[kx,ky,kz]])) 
    
    # fig = go.Figure(data=go.Volume(
    #     x=X.flatten(),
    #     y=Y.flatten(),
    #     z=Z.flatten(),
    #     value=(values).flatten(),
    #     isomin=0.1,
    #     isomax=1,
    #     opacity=0.2, # needs to be small to see through all surfaces
    #     surface_count=17, # needs to be a large number for good volume rendering
    #     ))
    # fig.show()

    if (total_savepoints>=lay.T):
        total_savepoints = lay.T
    avg_freq = lay.T/total_savepoints
    
    L = []
    for s in range(total_savepoints):
        L.append(lay.T - 1 - int(avg_freq*s))


    for k in range(K):
        print("  " + str(k + 1) + " / " + str(K) + "   ( " + str(np.round(100*k*1.0/K,2)) +" % ) .",end="\r")
        if (k==1000):
            verbose = True
        else : 
            verbose = False
        lay.verbose = verbose
        
        exp_sumup = layer_exp_sumup(k,datapath)
        #for run_comp in lay.run_generator():
        for layer in nf_generator(lay,f,g,np.array([0]),initial_observation=None):
            if ((lay.t-1) in L):
                tick_sumup = layer_tick_sumup(layer=lay,k=k,supplex=None,t = lay.t)
                exp_sumup.add_layer_tick_sumup(tick_sumup)
                # summary = layer_sumup(save_dir=datapath,layer=lay,k = k,supplex="test") # layer at time t, for run k
                # summary.save_base() # Save it to the correct file
        exp_sumup.save_layer_exp_sumup()

        # evals.append(evaluate_run(lay.efficacy, lay.o)[0])
        # if not(isField(lay.a_)):
        #     a_matrices.append([flexible_copy(lay.A_[0]),flexible_copy(lay.A_[0])])
        #     matrix_diff.append([matrix_distance(normalize(lay.A_[0]), lay.A_[0]),matrix_distance(normalize(lay.b_[0]), lay.B_[0])])
        # else :
        #     a_matrices.append([flexible_copy(lay.a_[0]),flexible_copy(lay.A_[0])])
        #     matrix_diff.append([matrix_distance(normalize(lay.a_[0]), lay.A_[0]),matrix_distance(normalize(lay.b_[0]), lay.B_[0])])
        # b_matrices.append([flexible_copy(lay.b_[0]),flexible_copy(lay.B_[0])])

        if(plotting_trials):
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

    if (plotting_trials):
        fig.show()

    if (anyplot):
        B = lay.B_[0]
        b = normalize(lay.b_[0],)
        multi_matrix_plot([B,b], ["Real B","Learnt B"])

        A = lay.A_[0]
        if not(isField(lay.a_)):
            a = A
        else :
            a = normalize(lay.a_[0])
        multi_matrix_plot([A,a], ["Real A","Learnt A"])

        fig,axes = plt.subplots()
        axes.plot(range(K),range(K),'o')
        fig.show()
        input()
    
    print(lay.u)
    
if (__name__ == "__main__"):
    
    
    A = np.array([[0,1,2,3,4],
                  [1,0,1,2,3],
                  [2,1,0,1,2],
                  [3,2,1,0,1],
                  [4,3,2,1,0]])

    B = np.array([[0],
                  [0],
                  [0],
                  [1],
                  [0]])

    C = np.array([[0],
                  [0],
                  [1],
                  [0],
                  [0]])
    
    B_right = np.zeros((5,5,5))
    B_right[:,:,0] = np.array([ [1,0,0,0,0],         # Stay at the same activity
                                [0,1,0,0,0],
                                [0,0,1,0,0],
                                [0,0,0,1,0],
                                [0,0,0,0,1]])

    B_right[:,:,1] = np.array([[0  ,0  ,0  ,0  ,0  ],         # Try to move to highrt activity
                              [1  ,0  ,0  ,0  ,0  ],
                              [0  ,1  ,0  ,0  ,0  ],
                              [0  ,0  ,1  ,0  ,0  ],
                              [0  ,0  ,0  ,1  ,1 ]])
    
    B_right[:,:,2] = np.array([[0  ,0  ,0  ,0  ,0  ],         # Move to higher activity (fast)
                              [0  ,0  ,0  ,0  ,0  ],
                              [1  ,0  ,0  ,0  ,0  ],
                              [0  ,1  ,0  ,0  ,0  ],
                              [0  ,0  ,1  ,1  ,1 ]])
    
    B_right[:,:,3] = np.array([[1  ,1  ,0  ,0  ,0  ],         # Move to lower activity
                              [0  ,0  ,1  ,0  ,0  ],
                              [0  ,0  ,0  ,1  ,0  ],
                              [0  ,0  ,0  ,0  ,1  ],
                              [0  ,0  ,0  ,0  ,0 ]])
    
    B_right[:,:,4] = np.array([[1  ,1  ,1  ,0  ,0  ],         # Move to lower activity (fast)
                              [0  ,0  ,0  ,1  ,0  ],
                              [0  ,0  ,0  ,0  ,1  ],
                              [0  ,0  ,0  ,0  ,0  ],
                              [0  ,0  ,0  ,0  ,0 ]])

    multi_matrix_plot([B,C],colmap='viridis',vmax=1)
    plt.show()
    # for k in range(10):
    #     run_name = "D:\\data\\neuromodels\\noise_run_2\\" + str("random_") + str(k) +"\\"
    #     custom_run(run_name, 300,anyplot=False,noiselevel = 1,sham=True)


    
    #simple_run_2("D:\\data\\neuromodels\\003_100\\",5000,anyplot=True)
    #custom_run("D:\\data\\neuromodels\\001\\",100,total_savepoints=1,anyplot=True)
    # #lay.run()
        # a = np.array([[0 , 0],
        #               [0 , 1],
        #               [1 , 0],
        #               [0 , 0],
        #               [0 , 0]])

        # print(dist_to_index(a))
        # print(index_to_dist(np.array([4,2]), a))