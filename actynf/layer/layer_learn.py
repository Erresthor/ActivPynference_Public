# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:44:10 2021

@author: Côme ANNICCHIARICO(come.annicchiarico@mines-paristech.fr), adaptation of the work of :

%% Step by step introduction to building and using active inference models

% Supplementary Code for: A Step-by-Step Tutorial on Active Inference Modelling and its 
% Application to Empirical Data

% By: Ryan Smith, Karl J. Friston, Christopher J. Whyte
(MATLAB Script)
https://github.com/rssmith33/Active-Inference-Tutorial-Scripts/blob/main/Step_by_Step_AI_Guide.m


AND 

Towards a computational (neuro)phenomenology of mental action: modelling
meta-awareness and attentional control with deep-parametric active inference (2021)

Lars Sandved-Smith  (lars.sandvedsmith@gmail.com)
Casper Hesp  (c.hesp@uva.nl)
Jérémie Mattout  (jeremie.mattout@inserm.fr)
Karl Friston (k.friston@ucl.ac.uk)
Antoine Lutz (antoine.lutz@inserm.fr)
Maxwell J. D. Ramstead (maxwell.ramstead@mcgill.ca)


------------------------------------------------------------------------------------------------------
A method initializing the various sizes used in Active Inference Experiments
"""
from operator import ne
import os,sys,time
import random as r
import time 
import numpy as np
import matplotlib.pyplot as plt

from ..base.miscellaneous_toolbox import flexible_copy
from ..base.extrapolate_diagonally import extrap_diag_2d
from ..base.function_toolbox import spm_dekron,spm_complete_margin,spm_cross,normalize
from .spm_backwards import spm_backwards,backward_state_posterior_estimation
from .utils import dist_from_definite_outcome_accross_t

from ..enums.memory_decay import MemoryDecayType
from ..enums.space_structure import AssumedSpaceStructure

class layerPlasticity:
    def __init__(self,eta,mem_decay_type,mem_loss,assume_state_space_structure,gen_f):
        self.eta = eta
        self.mem_dec_type = mem_decay_type
        self.t05 = mem_loss
        self.state_space_hypothesis = assume_state_space_structure  
        self.generalize_fadeout = gen_f


def update_rule(old_matrix,new_matrix,mem_dec_type,T,t05 = 100,eps1 = 1e-7,eps2=1e-8):
    """
    Update one's belief about environment dynamics given new observations / inferences.
    Affected by the memory decay term, which defines how much the new observation is worth 
    compared to previous beliefs.
    """
    if(mem_dec_type== MemoryDecayType.PROPORTIONAL):
        t05 = (T/2)  # Memory loss factor : guarantee that at the end of the experiment , only remain_percentage % of initial knowledge remain
    elif (mem_dec_type==MemoryDecayType.STATIC):
        t05 = t05
    elif(mem_dec_type==MemoryDecayType.NO_MEMORY_DECAY) :
        t05 = 0.0
    
    if (t05 <= eps1):
        return old_matrix + new_matrix
    else :
        multiplier = np.exp(-(np.log(2)/t05))

        new_matrix = old_matrix*multiplier + new_matrix
        new_matrix[new_matrix<eps2] = eps2
        return new_matrix
    
def generalize(base_information, structure_assumption,fadeout_function=(lambda x:1.0)):
    # print(structure_assumption)
    if (structure_assumption == AssumedSpaceStructure.NO_STRUCTURE):
        return base_information
    
    # If we assume a linear structure within this factor's state space
    if ("LINEAR" in structure_assumption.name) :
        clamp_interp = (structure_assumption==AssumedSpaceStructure.LINEAR_CLAMPED)
        periodic_interp = (structure_assumption==AssumedSpaceStructure.LINEAR_PERIODIC)
        generalized_information = extrap_diag_2d(base_information,clamp_interp,periodic_interp,
                                                 fadeout_function,True)
    
    return generalized_information

def a_learning(o_d_history,s_kron_d_history,old_a_matrix,
               plasticityOptions):
    """Returns an updated perception matrix given an history of :
    - observations o_d
    - state inferences s_d
    - old perception matrix a
    """
    Nmod = len(old_a_matrix)
    new_a = flexible_copy(old_a_matrix)
    
    T = s_kron_d_history.shape[-1]    
    for modality in range(Nmod):
        da = 0
        for t in range(T):   
            od_t = o_d_history[modality][:,t]
            xd_t = s_kron_d_history[:,t]
            da_t = spm_cross(od_t,xd_t)

            da_t = (np.reshape(da_t,old_a_matrix[modality].shape))
            da_t = da_t*(old_a_matrix[modality]>0)

            da = da + da_t
        new_a[modality] = update_rule(old_a_matrix[modality],da*plasticityOptions.eta,plasticityOptions.mem_dec_type,T,plasticityOptions.t05)
        #layer.a_[modality] = k*layer.a_[modality] + da*eta
    return new_a

def b_learning(u_d_history,s_margin_history,old_b_matrix,action_transition_mapping,
               plasticityOptions):
    """Returns an updated transition matrix given an history of :
    - state inferences across times s_d
    - action inferences across times u_d
    - old transition matrix b
    - action_transition_mapping :  U <-> How does the INDEX of the selected action
                                     inform us about the selected STATE transition
    """
    Nf = len(old_b_matrix)
    Ntransitions = [factor_b.shape[-1] for factor_b in old_b_matrix]
    T = u_d_history.shape[-1] + 1
    new_b = flexible_copy(old_b_matrix)

    # Timewise distribution of factorwise transitions selected
    transition_prob_matrix = []
    for factor in range(Nf):
        action_leads_to_transition_at_factor = action_transition_mapping[:,factor]
        transition_prob_matrix.append(np.zeros((Ntransitions[factor],T-1)))
        for t in range(T-1):
            transition_prob_matrix[factor][:,t] = np.bincount(action_leads_to_transition_at_factor,u_d_history[:,t])
            # Prob density of selected factorwise transitions between states. (the space is the possible transitions for 
            # a select state factor)
    
    for factor in range(Nf):
        if (type(plasticityOptions.state_space_hypothesis)==list):
            assumed_space_structure = plasticityOptions.state_space_hypothesis[factor]
        else : 
            assumed_space_structure = plasticityOptions.state_space_hypothesis

        db = 0
        for t in range(1,T):
            factor_action_implemented = transition_prob_matrix[factor][:,t-1]
            action_independent_transition =np.outer(s_margin_history[factor][:,t],s_margin_history[factor][:,t-1])
                    # Independently of actions, what is the perceived state transition that took place ?

            # If we entertain specific structural hypotheses regarding the hidden state space, 
            # we may learn more from a single observation !
            # assumed_space_structure = plasticityOptions.state_space_hypothesis
            generalized_action_independent_transition = generalize(action_independent_transition,assumed_space_structure,plasticityOptions.generalize_fadeout)

            db_t = spm_cross(generalized_action_independent_transition,factor_action_implemented)
                    # Compute the added B matrix : the cross product of the actual state evolution 
                    # and the probability density of the transition that took place.
            
            db_t = db_t*(old_b_matrix[factor]>0)
            db = db + db_t
        new_b[factor] = update_rule(old_b_matrix[factor],db*plasticityOptions.eta,plasticityOptions.mem_dec_type,T,plasticityOptions.t05)
    return new_b

def c_learning(o_d_history,old_c_matrix,plasticityOptions):
    raise NotImplementedError ("C_learning has not been implemented yet ...")

    # Nf = layer.Nf
    # Np = layer.Np
    # Nmod = layer.Nmod
    # T = layer.T

    # eta = layer.parameters.eta 

    # for modality in range(Nmod):
    #     dc = layer.O[modality][:,t]
    #     if (layer.c_[modality].shape[1]>1) : #If preferences are dependent on time
    #         dc = dc*(layer.c_[modality][:,t]>0)
    #         layer.c_[modality][:,t] = layer.c_[modality][:,t] + dc*layer.eta
    #     else : 
    #         dc = dc*(layer.c_[modality] > 0)
    #         layer.c_[modality] = layer.c_[modality] + dc*eta

def d_learning_base(o_margin_history,x_kron_history,a_kron,
                kronecker_transition_history,
                old_d_matrix,plasticityOptions):
    Nf = len(old_d_matrix)
    Ns = tuple([k.shape[0] for k in old_d_matrix])

    # print(o_margin_history[1])

    L = spm_backwards(o_margin_history, x_kron_history, a_kron, kronecker_transition_history)
    # print(L)
    dek = spm_dekron(L, Ns)

    new_d_matrix = flexible_copy(old_d_matrix)
    for factor in range(Nf):
        i = old_d_matrix[factor]>0
        #layer.d_[factor][i] = layer.d_[factor][i] + dek[factor][i]*layer.parameters.eta 
        new_d_matrix[factor][i] = update_rule(old_d_matrix[factor][i],plasticityOptions.eta*dek[factor][i],plasticityOptions.mem_dec_type,1,plasticityOptions.t05)
    return new_d_matrix

def d_learning_smooth(s_margin_history,
                old_d_matrix,plasticityOptions):
    Nf = len(old_d_matrix)

    new_d_matrix = flexible_copy(old_d_matrix)
    for factor in range(Nf):
        i = old_d_matrix[factor]>0
        d_estimate = s_margin_history[factor][:,0]
        #layer.d_[factor][i] = layer.d_[factor][i] + dek[factor][i]*layer.parameters.eta 
        new_d_matrix[factor][i] = update_rule(old_d_matrix[factor][i],plasticityOptions.eta*d_estimate,plasticityOptions.mem_dec_type,1,plasticityOptions.t05)
    return new_d_matrix

def e_learning(layer,t05 = 100):
    raise NotImplementedError ("E_learning has not been implemented yet ...")

def learn_from_experience(layer):
    show_timers = False
    t_first_learn = time.time()
    # print("----------------  LEARNING  ----------------")
    T = layer.T
    N = layer.T_horizon
    Nmod = layer.Nmod
    Nf = layer.Nf


    backwards_pass = layer.learn_options.backwards_pass
    eta = layer.learn_options.eta
    mem_loss = layer.learn_options.memory_loss
    mem_decay_type = layer.learn_options.decay_type
    
    assume_state_space_structure = layer.learn_options.assume_state_space_structure
    exponential_decay_function = (lambda x: np.exp(-layer.learn_options.generalize_fadeout_function_temperature*x))
    # print(assume_state_space_structure)
    if (type(assume_state_space_structure)==list):
        if not(len(assume_state_space_structure)==Nf):
            assert len(assume_state_space_structure)==1, "Your defined structural assumptions list length does not match the number of states : " + str(len(assume_state_space_structure)) + " =/= " + str(Nf)
            assume_state_space_structure = assume_state_space_structure[0]
            print("WARNING : Got one value in the list of space structure assumption. Assuming that this assumption is valid for all states factors.")
            print("If this is wanted behviour and you want to hide this message, please set your assumed state space structure to AssumedSpaceStructure type.")
    general_plasticity = layerPlasticity(eta,mem_decay_type,mem_loss,assume_state_space_structure,exponential_decay_function)

    STM = layer.STM

    o_history = STM.o
    o_d_history = STM.o_d
    
    x_history = STM.x
    x_d_history = STM.x_d
    
    t_first= time.time()
    x_kron_history = layer.joint_to_kronecker_accross_time(x_d_history)
    if show_timers :
        print("     Kroneckerization pass took {:.2f} seconds".format(time.time() - t_first))

    u_history = STM.u
    u_d_history = STM.u_d
    
    # LEARNING :
    t_first= time.time()
    marginalized_o = spm_complete_margin(o_d_history,o_d_history.ndim-1)
    marginalized_x = spm_complete_margin(x_d_history,x_d_history.ndim-1)
    if show_timers :
        print("     Marginalization pass took {:.2f} seconds".format(time.time() - t_first))
    # print(np.round(marginalized_o[0],2))

    using_realized_actions = True  # If there is ambiguity regarding the last selected action
                                   # Useful in contexts such as BCI ?
    if using_realized_actions:
        transition_history_u_d = dist_from_definite_outcome_accross_t(u_history,u_d_history.shape)
    else :
        transition_history_u_d = u_d_history[:,t]


    # This is a potentially very time consuming operation for large number of states
    # let's just sample here
    # # OLD CODE
    # t_first= time.time()
    # b_kron_action_model_avg = []
    # for t in range(T-1):
    #     b_kron_action_model_avg.append(layer.kronecker_action_model_average(transition_history_u_d[:,t]))
    # print("     Transition model averaging pass took {:.2f} seconds".format(time.time() - t_first))

    # NEW CODE
    t_first= time.time()
    b_kron_action_model_avg = []
    for t in range(T-1):
        b_kron_action_model_avg.append(layer.kronecker_action_model_average(u_history[t],just_slice=True))
    if show_timers :
        print("     Transition model averaging pass took {:.2f} seconds".format(time.time() - t_first))


    backwars_pass_is_fixed = False
    if backwards_pass and backwars_pass_is_fixed :
        t_first = time.time()
        smoothed_x_kron_history = backward_state_posterior_estimation(marginalized_o,x_kron_history,layer.var.a_kron,b_kron_action_model_avg)
        STM.x_d_smoothed = layer.kronecker_to_joint_accross_time(smoothed_x_kron_history) # Let's save it to the layer's STM !
        marginalized_smoothed_x = spm_complete_margin(STM.x_d_smoothed,x_d_history.ndim-1)
        if show_timers :
            print("     Backward pass took {:.2f} seconds".format(time.time() - t_first))
    else :
        smoothed_x_kron_history = x_kron_history
        marginalized_smoothed_x = marginalized_x
        # Nothing to save to the STM 
    # print(marginalized_smoothed_x)

    # fig,axs = plt.subplots(2)
    # axs[0].imshow(layer.a[0])
    if (layer.learn_options.learn_a): 
        t_first = time.time()
        new_a = a_learning(marginalized_o,smoothed_x_kron_history,layer.a,
                general_plasticity)
        layer.a = new_a
    #     if show_timers :
    #         print("     Learning a took {:.2f} seconds".format(time.time() - t_first))
    # axs[1].imshow(layer.a[0])
    # fig.show()
    

    # plot_Gd = False
    # if plot_Gd :
    #     action_select_hist = STM.Gd
    #     fig2,axs2= plt.subplots(1,T-1)

    #     axs2[0].set_ylabel("Actions")
    #     for t in range(T-1):
    #         im = axs2[t].imshow(action_select_hist[...,t])
    #         axs2[t].set_xticks(range(action_select_hist.shape[0]))
    #         axs2[t].set_xticklabels(["Habits","Exploit","Uncertainty","FB novelty","ACT novelty","Deeper"],rotation=45,fontsize=4)
    #         fig2.colorbar(im,fraction=0.046, pad=0.04)
    #     fig2.show()
        
    #     input()

    if (layer.learn_options.learn_b):
        t_first = time.time()
        new_b = b_learning(transition_history_u_d,marginalized_smoothed_x,layer.b,layer.U,
               general_plasticity) 
        layer.b = new_b
        if show_timers :
            print("     Learning b took {:.2f} seconds".format(time.time() - t_first))

    if (layer.learn_options.learn_c) :
        c_learning(marginalized_o,layer.c,
                general_plasticity)
    
    if (layer.learn_options.learn_d) : #Update initial hidden states beliefs
        t_first = time.time()
        if (layer.learn_options.use_backward_pass_to_learn_d):
            # print("D Learning boi")
            new_d = d_learning_smooth(marginalized_smoothed_x,layer.d,general_plasticity)
        else :
            new_d = d_learning_base(marginalized_o,x_kron_history,layer.var.a_kron,
                            b_kron_action_model_avg,layer.d,
                    general_plasticity)
        layer.d = new_d   
        if show_timers :
            print("     Learning d took {:.2f} seconds".format(time.time() - t_first))
    
    if show_timers :
        print("Learning everything for " + layer.name + " took {:.2f} seconds".format(time.time() - t_first_learn))

    # if (layer.learn_options.learn_e): # Update agent habits
    #     e_learning(layer,t05 = t05)

    # if (Np>1):
    #     dn = 8*np.gradient(layer.wn) + layer.wn/8.0
    # else :
    #     dn = None
    #     wn = None
    
    # Xn = []
    # Vn = []
    # # BMA Hidden states
    # for factor in range(Nf):
    #     Xn.append(np.zeros((Ni,Ns[factor],T,T)))
    #     Vn.append(np.zeros((Ni,Ns[factor],T,T)))
        
    #     for t in range(T):
    #         for policy in range(Np):
    #             Xn[factor][:,:,:,t] = Xn[factor][:,:,:,t] + np.dot(xn[factor][:,:,:,t,policy],u[policy,t])
    #             Vn[factor][:,:,:,t] = Vn[factor][:,:,:,t] + np.dot(vn[factor][:,:,:,t,policy],u[policy,t])
    # print("Learning and encoding ended without errors.")
    
    # if isField(layer.U_):
    #     u = u[:,:-1]
    #     un =  un[:,:-Ni]

""" 
When learning new dynamics, observations can be interpreted as 
evidence for some environment representation change. (e.g. i've 
seen a red shape with wheels and i've infered it was a car, it will make me 
learn that red objects  as well as wheeled objects may often be cars)

"""