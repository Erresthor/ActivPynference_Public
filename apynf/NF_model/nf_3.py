#!/bin/python
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
import plotly.graph_objects as go

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from base.spm12_implementation import MDP 
from base.miscellaneous_toolbox import flexible_copy , isField , index_to_dist, dist_to_index
from base.function_toolbox import normalize
from mdp_layer import mdp_layer
from neurofeedback_base import NF_model_displayer
from base.plotting_toolbox import multi_matrix_plot
from base.file_toolbox import load_flexible,save_flexible
import matplotlib.pyplot as plt

from base.function_toolbox import spm_dot,spm_kron

from layer_postrun import evaluate_run
from layer_learn import MemoryDecayType
from pynf_functions import *
from file_toolbox import root_path
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
    initial_state = 0



    print("Explore-Exploit --- Model set-up ...  ",end='')
    #Points within a trial
    
    Nf = 1

    # Priors about initial states
    # Prior probabilities about initial states in the generative process
    D_ =[]
    # Context state factor
    D_.append(np.array([0,0,0,0,0])) #[Terrible state, neutral state , good state, great state, excessive state]
    D_[0][initial_state] = 1
    
    # Prior beliefs about initial states in the generative process
    d_ =[]
    # Mental states
    d_.append(np.array([0.996,0.001,0.001,0.001,0.001])) #[Terrible state, neutral state , good state, great state, excessive state]

    # State Outcome mapping and beliefs
    # Prior probabilities about initial states in the generative process
    Ns = [D_[0].shape[0]] #(Number of states)
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
    pa = 1
    A_obs_mental[:,:] = np.array([[pa  ,0.5-0.5*pa,0         ,0         ,0   ],
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
        a_[0][i,i] = 10

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
    b_ = []
    for fac in range (len(B_)):
        b_.append(np.copy(B_[fac])*100)
    b_[0] = np.ones((b_[0].shape))*1
    # b_[0][1,:,:] = 0.15
    # b_[0][2,:,:] = 0.2
    # b_[0][3,:,:] = 0.3
    # b_[0][4,:,:] = 0.4

    # for i in range(B_[0].shape[-1]):
    #     b_[0][:,:,i][B_[0][:,:,i]>0.5] += 10
    
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
    Nf = 1 #Number of state factors


    U_ = np.zeros((nu,Nf)).astype(np.int)
    U_[:,0] = range(nu)
    
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
    
    #model.E_ = E_
    #model.e_ = e_
    
    #Other parameters
    print("Done")
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
        B_mental_states = layer.B_[0]

        posterior_dist = B_mental_states[:,previous_state,action_chosen]

        next_state_dist = np.argwhere(r.random() <= np.cumsum(posterior_dist,axis=0))[0][0]

        return np.array([next_state_dist])

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


def main(foldername:str,noiselevel,T:int=1500,same:int = 10,sham=False):
    
    
    path = os.path.join(root_path(),"/mnt/data/Come_A/data/")

    complete_foldername = os.path.join(path,foldername)

    print(complete_foldername)
    for k in range(same):
        run_name = os.path.join(str(round(noiselevel,2)), str(k))

        complete_run_name = os.path.join(complete_foldername,run_name)

        if (os.path.isdir(run_name)) :
            print(run_name  + "  already exists. Skipping this simulation.")
        else:
            print("Executing simulation " + str(k) + " / " + str(same) + " for nl = " + str(noiselevel) + " at " + run_name)
            custom_run(complete_run_name, T,anyplot=False,noiselevel = noiselevel)
        
        print('-------------------------------------')


if (__name__ == "__main__"):
    foldername = str(sys.argv[1])
    noiselevel = float(sys.argv[2])
    T = int(sys.argv[3])
    same = int(sys.argv[4])
    sham = (str(sys.argv[5]).lower()=="true")

    main(foldername,noiselevel,T=T,same=same,sham=sham)


# if (__name__ == "__main__"):
#     N = 50
#     L = np.linspace(0,1,N)


#     for run in range(N):
#         main("neuro_test_4",L[run],T=1500,same=10)