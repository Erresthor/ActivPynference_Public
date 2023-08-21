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
import os,sys
from enum import Enum
import random as r
import time 
import numpy as np
import matplotlib.pyplot as plt

from ..base.miscellaneous_toolbox import isField,flexible_copy,flatten_last_n_dimensions
from ..base.function_toolbox import normalize , spm_wnorm, nat_log , spm_psi, softmax , spm_cross
from ..base.function_toolbox import spm_kron,spm_margin,spm_dekron,spm_KL_dir,spm_complete_margin
from .spm_backwards import spm_backwards,backward_state_posterior_estimation
from .utils import dist_from_definite_outcome_accross_t

# MEMORY DECAY MECHANICS : 
class MemoryDecayType(Enum):
    NO_MEMORY_DECAY = 0
    PROPORTIONAL = 1
    STATIC = 2

def update_rule(old_matrix,new_matrix,mem_dec_type,T,t05 = 100):
    if(mem_dec_type== MemoryDecayType.PROPORTIONAL):
        t05 = (T/2)  # Memory loss factor : guarantee that at the end of the experiment , only remain_percentage % of initial knowledge remain
    elif (mem_dec_type==MemoryDecayType.STATIC):
        t05 = t05
    elif(mem_dec_type==MemoryDecayType.NO_MEMORY_DECAY) :
        t05 = 0.0
    
    eps = 1e-7
    if (t05 <= eps):
        return old_matrix + new_matrix
    else :
        epsilon = 1e-8
        k = 1
        multiplier = np.exp(-(np.log(2)/t05))

        new_matrix = old_matrix*multiplier + new_matrix
        new_matrix[new_matrix<epsilon] = epsilon
        return new_matrix
    
class layerPlasticity:
    def __init__(self,eta,mem_loss):
        if (mem_loss>0):
            self.t05 = mem_loss
            self.mem_dec_type = MemoryDecayType.STATIC
        else : 
            self.t05 = 0.0
            self.mem_dec_type = MemoryDecayType.NO_MEMORY_DECAY
        
        self.eta = eta

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
    
    for factor in range(Nf):
        db = 0
        for t in range(1,T):
            factor_action_implemented = transition_prob_matrix[factor][:,t-1]
            action_independent_transition =np.outer(s_margin_history[factor][:,t],s_margin_history[factor][:,t-1])
            db_t = spm_cross(action_independent_transition,factor_action_implemented)
            db_t = db_t*(old_b_matrix[factor]>0)
            
            db = db + db_t
        new_b[factor] = update_rule(old_b_matrix[factor],db*plasticityOptions.eta,plasticityOptions.mem_dec_type,T,plasticityOptions.t05)
    return new_b

def c_learning(o_d_history,old_c_matrix,plasticityOptions):
    print("C LEARNING :")
    print("This function has not been implemented yet ...")
    return
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

def d_learning(o_margin_history,x_kron_history,a_kron,
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

def e_learning(layer,t05 = 100):
    T = layer.T
    de = 0
    eta = layer.parameters.eta 

    for t in range(T-1) :
        de = de + layer.u[:,t]
    layer.e_ = layer.e_ + de*eta 

def learn_from_experience(layer):
    # print("----------------  LEARNING  ----------------")
    T = layer.T
    N = layer.T_horizon
    Nmod = layer.Nmod
    Nf = layer.Nf


    backwards_pass = layer.learn_options.backwards_pass
    eta = layer.learn_options.eta
    mem_loss = layer.learn_options.memory_loss
    general_plasticity = layerPlasticity(eta,mem_loss)

    STM = layer.STM
    o_history = STM.o
    o_d_history = STM.o_d
    
    x_history = STM.x
    x_d_history = STM.x_d
    x_kron_history = layer.joint_to_kronecker_accross_time(x_d_history)
    
    u_history = STM.u
    u_d_history = STM.u_d
    
    # LEARNING :
    marginalized_o = spm_complete_margin(o_d_history,o_d_history.ndim-1)
    marginalized_x = spm_complete_margin(x_d_history,x_d_history.ndim-1)
    # print(np.round(marginalized_o[0],2))

    using_realized_actions = True
    if using_realized_actions:
        transition_history_u_d = dist_from_definite_outcome_accross_t(u_history,u_d_history.shape)
    else :
        transition_history_u_d = u_d_history[:,t]

    
    b_kron_action_model_avg = []
    for t in range(T-1):
        # print(transition_history_u_d[:,t])
        # print(u_d_history[:,t])
        b_kron_action_model_avg.append(layer.kronecker_action_model_average(transition_history_u_d[:,t]))
    
    # print(np.round(x_kron_history,2))
    
    if backwards_pass :
        smoothed_x_kron_history = backward_state_posterior_estimation(marginalized_o,x_kron_history,layer.var.a_kron,b_kron_action_model_avg)
        marginalized_smoothed_x = spm_complete_margin(layer.kronecker_to_joint_accross_time(smoothed_x_kron_history),x_d_history.ndim-1)
        # Let's save it to the layer's STM !
        STM.x_d_smoothed = smoothed_x_kron_history
    else :
        smoothed_x_kron_history = x_kron_history
        marginalized_smoothed_x = marginalized_x
        # Nothing to save to the STM 

    if (layer.learn_options.learn_a): 
        new_a = a_learning(marginalized_o,smoothed_x_kron_history,layer.a,
                general_plasticity)
        layer.a = new_a
    

    if (layer.learn_options.learn_b):
        new_b = b_learning(transition_history_u_d,marginalized_smoothed_x,layer.b,layer.U,
               general_plasticity) 
        layer.b = new_b

    if (layer.learn_options.learn_c) :
        c_learning(marginalized_o,layer.c,
                general_plasticity)

    if (layer.learn_options.learn_d) : #Update initial hidden states beliefs
        new_d = d_learning(marginalized_o,x_kron_history,layer.var.a_kron,
                           b_kron_action_model_avg,layer.d,
                general_plasticity)
        layer.d = new_d   
        
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