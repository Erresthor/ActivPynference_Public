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


from ..base.miscellaneous_toolbox import flexible_copy , isField , index_to_dist, dist_to_index
from ..base.function_toolbox import normalize
from ..base.plotting_toolbox import multi_matrix_plot
from ..base.file_toolbox import load_flexible,save_flexible
from ..base.function_toolbox import spm_dot,spm_kron
from ..base.matrix_functions import *
from ..base.file_toolbox import root_path
from ..layer_old.mdp_layer import mdp_layer
from ..layer_old.layer_postrun import evaluate_run
from ..layer_old.layer_learn import MemoryDecayType
from ..layer_old.layer_sumup import *

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
    b_[0][1,:,:] = 0.15
    b_[0][2,:,:] = 0.2
    b_[0][3,:,:] = 0.3
    b_[0][4,:,:] = 0.4

    for i in range(B_[0].shape[-1]):
        b_[0][:,:,i][B_[0][:,:,i]>0.5] += 10
    
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
    layer.options.decay_half_time = 100


    layer.Ni = 16
    layer.A_ = A_
    #layer.a_ = a_
    
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

def generate_optimal_observations(layer):
    obs = np.zeros((1,layer.T))
    o = 0
    for t in range(layer.T):
        obs[0,t] = o
        if(o<layer.C_[0].shape[0]-1):
            o = o +1
    layer.prep_trial()
    return evaluate_run(layer.efficacy, obs.astype(np.int))

def simple_run(datapath,K,total_savepoints = 10,anyplot=False,plotting_trials = False,max_per_line = 8):
    lay = nf_2_model(5,-2)
    lay.options.learn_during_experience = True #false

    if (anyplot):
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
    
    if (total_savepoints>=lay.T):
        total_savepoints = lay.T
       
    avg_freq = lay.T/total_savepoints
    
    L = []
    for s in range(total_savepoints):
        L.append(lay.T - 1 - int(avg_freq*s))
    print(L)

    evals = []
    matrix_diff=[]
    a_matrices = []
    b_matrices = []



    for k in range(K):
        print(k)

        if (k==10):
            verbose = True
        else : 
            verbose = False
        lay.verbose = verbose

        exp_sumup = layer_exp_sumup(k,datapath)
        for run_comp in lay.run_generator():
            if ((lay.t-1) in L):
                tick_sumup = layer_tick_sumup(layer=lay,k=k,supplex=None,t = lay.t)
                exp_sumup.add_layer_tick_sumup(tick_sumup)
                # summary = layer_sumup(save_dir=datapath,layer=lay,k = k,supplex="test") # layer at time t, for run k
                # summary.save_base() # Save it to the correct file
        exp_sumup.save_layer_exp_sumup()

        evals.append(evaluate_run(lay.efficacy, lay.o)[0])
        matrix_diff.append([matrix_distance(normalize(lay.a_[0]), lay.A_[0]),matrix_distance(normalize(lay.b_[0]), lay.B_[0])])
        a_matrices.append([flexible_copy(lay.a_[0]),flexible_copy(lay.A_[0])])
        b_matrices.append([flexible_copy(lay.b_[0]),flexible_copy(lay.B_[0])])

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

        #print(evaluate_run(lay.efficacy, lay.o))


        a_diff = lay.a[0] - exp_sumup.get_final_state().a_[0]
        if(np.sum(a_diff) > 0.01):
            print(k)
        b_diff = lay.b[0] - exp_sumup.get_final_state().b_[0]
        if(np.sum(b_diff) > 0.01):
            print(k)
        o_diff = lay.o - exp_sumup.get_final_state().o
        if(np.sum(o_diff) > 0.01):
            print(k)
        #print(lay.efficacy)
        eff_diff = lay.efficacy[0] - exp_sumup.get_final_state().efficacy[0]
        if(np.sum(a_diff) > 0.01):
            print(k)






        # print(matrix_diff[k])
        # print(evals[k])
        # print("-----")
    if (plotting_trials):
        fig.show()
    # lay.run()

    basepath = "D:\\data\\"+ "test\\"

    path = basepath + "evals_1.txt"
    save_flexible(evals, path)
    path = basepath + "diffs.txt"
    save_flexible(matrix_diff, path)  
    path = basepath + "a_1.txt"
    save_flexible(a_matrices, path)
    path = basepath + "b_1.txt"
    save_flexible(b_matrices, path)  

    if (anyplot):
        B = lay.B_[0]
        b = normalize(lay.b_[0],)
        multi_matrix_plot([B,b], ["Real B","Learnt B"])

        A = lay.A_[0]
        a = normalize(lay.a_[0])
        multi_matrix_plot([A,a], ["Real A","Learnt A"])

        fig,axes = plt.subplots()
        axes.plot(range(K),evals,'o')
        fig.show()
        print(lay.u)
        input()

def simple_run_2(datapath,K,total_savepoints = 10,anyplot=False,plotting_trials = False,max_per_line = 8):
    def change_B(B):
        new_B = flexible_copy(B)

        for i in range(B[0].shape[-1]) :
            new_B[0][:,:,i] = B[0][:,:,B[0].shape[-1]-i-1]

        return new_B



    lay = nf_2_model(5,-2)
    lay.options.learn_during_experience = True #false

    if (anyplot):
        B = lay.B_[0]
        b = normalize(lay.b_[0],)
        multi_matrix_plot([B,b], ["Real B","Prior b"])

        A = lay.A_[0]
        if not(isField(lay.a_)):
            a = A
        else :
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
    
    if (total_savepoints>=lay.T):
        total_savepoints = lay.T
       
    avg_freq = lay.T/total_savepoints
    
    L = []
    for s in range(total_savepoints):
        L.append(lay.T - 1 - int(avg_freq*s))
    print(L)

    evals = []
    matrix_diff=[]
    a_matrices = []
    b_matrices = []



    for k in range(K):
        print(k)

        if (k==10):
            verbose = True
        else : 
            verbose = False
        lay.verbose = verbose

        exp_sumup = layer_exp_sumup(k,datapath)
        for run_comp in lay.run_generator():
            if ((lay.t-1) in L):
                tick_sumup = layer_tick_sumup(layer=lay,k=k,supplex=None,t = lay.t)
                exp_sumup.add_layer_tick_sumup(tick_sumup)
                # summary = layer_sumup(save_dir=datapath,layer=lay,k = k,supplex="test") # layer at time t, for run k
                # summary.save_base() # Save it to the correct file
        exp_sumup.save_layer_exp_sumup()

        evals.append(evaluate_run(lay.efficacy, lay.o)[0])
        if not(isField(lay.a_)):
            a_matrices.append([flexible_copy(lay.A_[0]),flexible_copy(lay.A_[0])])
            matrix_diff.append([matrix_distance(normalize(lay.A_[0]), lay.A_[0]),matrix_distance(normalize(lay.b_[0]), lay.B_[0])])
        else :
            a_matrices.append([flexible_copy(lay.a_[0]),flexible_copy(lay.A_[0])])
            matrix_diff.append([matrix_distance(normalize(lay.a_[0]), lay.A_[0]),matrix_distance(normalize(lay.b_[0]), lay.B_[0])])
        b_matrices.append([flexible_copy(lay.b_[0]),flexible_copy(lay.B_[0])])

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

        if (k==int(K/2)) :
            lay.B_ = change_B(lay.B_)

        #print(evaluate_run(lay.efficacy, lay.o))

        if not(isField(lay.a_)):
            a_diff = np.zeros(lay.A_[0].shape)
        else :
            a_diff = lay.a[0] - exp_sumup.get_final_state().a_[0]
        
        if(np.sum(a_diff) > 0.01):
            print(k)
        b_diff = lay.b[0] - exp_sumup.get_final_state().b_[0]
        if(np.sum(b_diff) > 0.01):
            print(k)
        o_diff = lay.o - exp_sumup.get_final_state().o

        eff_diff = lay.efficacy[0] - exp_sumup.get_final_state().efficacy[0]


    if (plotting_trials):
        fig.show()

    basepath = "D:\\data\\"+ "test\\"

    path = basepath + "evals_1.txt"
    save_flexible(evals, path)
    path = basepath + "diffs.txt"
    save_flexible(matrix_diff, path)  
    path = basepath + "a_1.txt"
    save_flexible(a_matrices, path)
    path = basepath + "b_1.txt"
    save_flexible(b_matrices, path)  

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
        axes.plot(range(K),evals,'o')
        fig.show()
        print(lay.u)
        print(normalize(lay.b_))
        input()

def nf_2_run():
    simple_run_2("D:\\data\\neuromodels\\003_100\\",5000,anyplot=True)


if (__name__ == "__main__"):
    simple_run_2("D:\\data\\neuromodels\\003_100\\",5000,anyplot=True)
    #custom_run("D:\\data\\neuromodels\\001\\",100,total_savepoints=1,anyplot=True)
    # #lay.run()
        # a = np.array([[0 , 0],
        #               [0 , 1],
        #               [1 , 0],
        #               [0 , 0],
        #               [0 , 0]])

        # print(dist_to_index(a))
        # print(index_to_dist(np.array([4,2]), a))