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
import os,sys
from enum import Enum
import random as r
import time 
import numpy as np
import matplotlib.pyplot as plt

from .parameters.policy_method import Policy_method
from ..base.miscellaneous_toolbox import isField
from ..base.function_toolbox import normalize , spm_wnorm, nat_log , spm_psi, softmax , spm_cross
from ..base.function_toolbox import spm_kron,spm_margin,spm_dekron,spm_KL_dir
from .spm_backwards import spm_backwards

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
        k = 1
        multiplier = np.exp(-(np.log(2)/t05))
        epsilon = k*(multiplier)
        min_value = 1
        return_knowledge_matrix = old_matrix
        return_knowledge_matrix[return_knowledge_matrix>min_value] = return_knowledge_matrix[return_knowledge_matrix>min_value] - (return_knowledge_matrix[return_knowledge_matrix>min_value]-min_value)*(1-epsilon)
        return_knowledge_matrix[return_knowledge_matrix<min_value] = min_value # Just in case ?

        return_knowledge_matrix = return_knowledge_matrix + new_matrix

        # # Problem with memory decay :
        # # If a specific distribution is NOT impossible BUT very low weigths
        # # (eg. a = [1e-5, 1e-9, 1e-9, 1e-9, 1e-9])
        # # the multiplier can cause some terms to go to 0 due to computational approxs
        # # (eg. a = [1e-5,    0,    0,    0,     0])
        # # This is not that much of a problem usually (if no paradigm changes)
        # # but if the multiplier keeps affecting the only remaining term, the distribution will become :
        # # (eg. a = [   0,    0,    0,    0,     0])
        # # Leading to a renormalization and a spike in uncertainty and error --> unwanted behaviour
        # # Therefore, we want to prevent such a behaviour by checking if any term should get to 0 
        # # If it is the case, we do not apply the multiplier to this line anymore ? No
        # # We keep the same relative weights and multiply their "strength" by a fixed factor K : "auto-reupper"
        # K = 2
        # epsilon = 1e-15 # same value used in normalize function. If below, normalize shows unwanted behaviour
        # is_too_low = (np.sum(return_knowledge_matrix,axis=-1)<=epsilon) 
        # # A distribution is too low if the sum of its terms is below eps
        
        # is_too_low = (np.min(return_knowledge_matrix,axis=-1)<=epsilon)&((np.min(return_knowledge_matrix,axis=-1)>0))
        # # A distribution is too low if the minimum of its non null terms is below eps
        # # If there is a weight below the threshold, the entire line is multiplied by K to prevent it from going below reinitialization thresh
        # # The "0" solution is a dirty trick to account for certain matrices that shouldn't be updated.
        
        # #print(np.min(return_knowledge_matrix,axis=-1))
        # return_knowledge_matrix[is_too_low,:] = return_knowledge_matrix[is_too_low,:]*K
        
        # # Check line by line if only one 
        return  return_knowledge_matrix





def a_learning(layer,t,mem_dec_type = MemoryDecayType.PROPORTIONAL,t05 = 100):
    Nmod = layer.Nmod
    Nf = layer.Nf
    T = layer.T

    eta = layer.parameters.eta

    for modality in range(Nmod):
        da = (layer.O[modality][:,t])
        if(layer.policy_method==Policy_method.ACTION):
            da = spm_cross(da,layer.Q[t])
        else: 
            for factor in range(Nf):
                da = spm_cross(da,layer.X[factor][:,t])
        #print(spm_dekron(da,tuple(layer.a_[modality].shape)))
        da = (np.reshape(da,layer.a_[modality].shape))
        da = da*(layer.a_[modality]>0)

        layer.a_[modality] = update_rule(layer.a_[modality],da*eta,mem_dec_type,T,t05)
        #layer.a_[modality] = k*layer.a_[modality] + da*eta

def b_learning(layer,t,mem_dec_type = MemoryDecayType.NO_MEMORY_DECAY,t05 = 100):
    Nf = layer.Nf
    Np = layer.Np
    Nmod = layer.Nmod
    T = layer.T

    eta = layer.parameters.eta 

    
    # Custom implementation of b learning :
    def output_action_probability_density(chosen_actions,b):
        output = []
        for factor in range(len(b)):
            output.append(np.zeros((chosen_actions.shape[-1],b[factor].shape[2])))
            # output.append(np.zeros((chosen_actions.shape[-1],b[factor].shape[1])))  # Size = T-1 x Ns[factor]
            for t in range(output[factor].shape[0]):
                output[factor][t,chosen_actions[factor,t]]=1
        return output
    

    action_probability_density = output_action_probability_density(layer.u,layer.b_)

    db = []
    for f in range(Nf):
        db.append(np.zeros(layer.b[f].shape))
    
    for factor in range(Nf):
        for policy in range(Np):
            if (layer.policy_method ==Policy_method.POLICY):
                u = layer.V[t-1,policy,factor]  # Action corresponding to currently examined policy   
                action_implemented = (action_probability_density[factor][t-1,u])  # Was this action implemented ? 1 (yes) / 0 (no)
            elif (layer.policy_method ==Policy_method.ACTION):
                u = layer.U[policy,factor]  # Action corresponding to currently examined policy       
                action_implemented = (action_probability_density[factor][t-1,u])  # Was this action implemented ? 1 (yes) / 0 (no)  

            transition_for_policy = np.outer(layer.x[factor][:,t,policy],layer.x[factor][:,t-1,policy])
                    # The following transition is expected to have happenned during this time
                    # if this policy was followed
                    # Column = To
                    # Line = From
                    # 3rd dim = Upon which action ?
            db[factor][:,:,u] = db[factor][:,:,u] + action_implemented*transition_for_policy
    
    for factor in range (Nf):
        db[factor] = db[factor]*(layer.b_[factor]>0)
        db[factor] = db[factor]/np.sum(db[factor])
        layer.b_[factor] = update_rule(layer.b_[factor],eta*db[factor],mem_dec_type,T,t05)

def c_learning(layer,t,mem_dec_type = MemoryDecayType.NO_MEMORY_DECAY,t05 = 100):
    Nf = layer.Nf
    Np = layer.Np
    Nmod = layer.Nmod
    T = layer.T

    eta = layer.parameters.eta 

    for modality in range(Nmod):
        dc = layer.O[modality][:,t]
        if (layer.c_[modality].shape[1]>1) : #If preferences are dependent on time
            dc = dc*(layer.c_[modality][:,t]>0)
            layer.c_[modality][:,t] = layer.c_[modality][:,t] + dc*layer.eta
        else : 
            dc = dc*(layer.c_[modality] > 0)
            layer.c_[modality] = layer.c_[modality] + dc*eta

def d_learning(layer,mem_dec_type = MemoryDecayType.NO_MEMORY_DECAY,t05 = 100):
    Nf = layer.Nf
    T = layer.T
    
    eta = layer.parameters.eta 

    L = spm_backwards(layer.O, layer.Q, layer.a, layer.b_kron, layer.K,layer.T)

    dek = spm_dekron(L, layer.Ns)
    
    print("###########################")
    print(dek)
    print("###########################")
    for factor in range(Nf):
        i = layer.d_[factor]>0
        #layer.d_[factor][i] = layer.d_[factor][i] + dek[factor][i]*layer.parameters.eta 
        layer.d_[factor][i] = update_rule(layer.d_[factor][i],layer.parameters.eta*dek[factor][i],mem_dec_type,T,t05)

def e_learning(layer,t05 = 100):
    T = layer.T
    de = 0
    eta = layer.parameters.eta 

    for t in range(T-1) :
        de = de + layer.u[:,t]
    layer.e_ = layer.e_ + de*eta 

def learn_from_experience(layer,mem_dec_type=MemoryDecayType.PROPORTIONAL,t05 = 100):
    print("----------------  LEARNING  ----------------")
    #print("Wow this was insightful : i'm gonna learn from that !")
    T = layer.T
    N = layer.options.T_horizon
    Nmod = layer.Nmod
    Nf = layer.Nf
    
    # LEARNING :
    for t in range(T):
        if isField(layer.a_)and(layer.options.learn_a): 
            a_learning(layer, t,mem_dec_type=mem_dec_type,t05 = t05)
                
        if isField(layer.b_)and (t>0) and(layer.options.learn_b):
            b_learning(layer, t,mem_dec_type,t05 = t05)
                    
        if isField(layer.c_)and(layer.options.learn_c) :
            c_learning(layer, t,mem_dec_type,t05 = t05)
        
    if isField(layer.d_)and(layer.options.learn_d) : #Update initial hidden states beliefs
        d_learning(layer,mem_dec_type,t05 = t05)
        
    if isField(layer.e_) and(layer.options.learn_e): # Update agent habits
        e_learning(layer,t05 = t05)
    
    # # Negative freeee eneergiiiies
    
    
    
    
    for modality in range (Nmod):
        if isField(layer.a_):
            Fa = -spm_KL_dir(layer.a_[modality],layer.a_prior[modality])
            layer.FE_dict['Fa'].append(Fa)
        if isField(layer.c_) :
            Fc = - spm_KL_dir(layer.c_[modality],layer.c_prior[modality])
            layer.FE_dict['Fc'].append(Fc)
    for factor in range(Nf):
        if isField(layer.b_):
            Fb = -spm_KL_dir(layer.b_[factor],layer.b_prior[factor])
            layer.FE_dict['Fb'].append(Fb)
        if isField(layer.d_):
            Fd = -spm_KL_dir(layer.d_[factor],layer.d_prior[factor])
            layer.FE_dict['Fd'].append(Fd)
    
    if (isField(layer.e_)):
        Fe = -spm_KL_dir(layer.e_,layer.e_prior)
        layer.FE_dict['Fe'].append(Fe)
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

def learn_during_experience(layer,ratio = 0.5):
    #print("Wow this was insightful : i'm gonna learn from that !")
    Nmod = layer.Nmod
    No = layer.No

    Nf = layer.Nf
    Ns = layer.Ns 

    Np = layer.Np
    Nu = layer.Nu

    Ni = layer.options.Ni
    current_t = layer.t
    T = layer.T

    N = layer.options.T_horizon

    # LEARNING :
    if isField(layer.a_): 
        a_learning(layer, current_t,layer.parameters.eta*ratio,mem_dec_type=MemoryDecayType.NO_MEMORY_DECAY,t05=0) # We learn at a reduced rate during experience, and without loss of information
            
    if isField(layer.b_)and (current_t>0) :
        b_learning(layer, current_t,layer.parameters.eta*ratio,mem_dec_type=MemoryDecayType.NO_MEMORY_DECAY,t05=0)  # We learn at a reduced rate during experience, and without loss of informatio