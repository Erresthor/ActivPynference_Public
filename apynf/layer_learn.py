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

from base.miscellaneous_toolbox import isField
from base.function_toolbox import normalize , spm_kron, spm_wnorm, nat_log , spm_psi, softmax , spm_cross
from parameters.policy_method import Policy_method
from base.function_toolbox import spm_kron,spm_margin,spm_dekron

def a_learning(layer,t):
    Nmod = layer.Nmod
    Nf = layer.Nf
    T = layer.T


    remain_percentage = 0.9
    k = np.exp((1/T)*np.log(remain_percentage))

    
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
        layer.a_[modality] = k*layer.a_[modality] + da*layer.parameters.eta

def b_learning(layer,t):
    Nf = layer.Nf
    Np = layer.Np
    T = layer.T

    remain_percentage = 0.9
    k = np.exp((1/T)*np.log(remain_percentage))

    # Custom implementation of b learning :
    def output_action_probability_density(chosen_actions,b):
        output = []
        for factor in range(len(b)):
            output.append(np.zeros((chosen_actions.shape[-1],b[factor].shape[1])))  # Size = T-1 x Np
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
            # print(t,factor,policy)
            # # print(layer.X[factor])
            # print(layer.x[factor][:,t,policy])
            # print(layer.x[factor][:,t-1,policy])
            # print(transition_for_policy)
            # print('--------------------')
            db[factor][:,:,u] = db[factor][:,:,u] + action_implemented*transition_for_policy
    
    for factor in range (Nf):
        db[factor] = db[factor]*(layer.b_[factor]>0)
        db[factor] = db[factor]/np.sum(db[factor])
        layer.b_[factor] =  k*layer.b_[factor] + layer.parameters.eta*db[factor]

def c_learning(layer,t):
    for modality in range(Nmod):
        dc = O[modality][:,t]
        if (layer.c_[modality].shape[1]>1) : #If preferences are dependent on time
            dc = dc*(layer.c_[modality][:,t]>0)
            layer.c_[modality][:,t] = layer.c_[modality][:,t] + dc*layer.eta
        else : 
            dc = dc*(layer.c_[modality] > 0)
            layer.c_[modality] = layer.c_[modality] + dc* layer.eta

def d_learning(layer):
    Nf = layer.Nf
    for factor in range (Nf):
        i = layer.d_[factor]>0
        layer.d_[factor][i] = layer.d_[factor][i] + layer.X[factor][i,0]

def e_learning(layer):
    T = layer.T
    layer.e_ = layer.e_ + layer.u[:,T-1]*layer.eta

def learn_from_experience(layer,reinitialise_afterwards = False):
    print("Wow this was insightful : i'm gonna learn from that !")
    Nmod = layer.Nmod
    No = layer.No

    Nf = layer.Nf
    Ns = layer.Ns 

    Np = layer.Np
    Nu = layer.Nu

    Ni = layer.options.Ni
    t_final = layer.t
    T = layer.T

    N = layer.options.T_horizon

    # LEARNING :
    for t in range(T):
        if isField(layer.a_): 
            a_learning(layer, t)
                
        if isField(layer.b_)and (t>0) :
            b_learning(layer, t)
                    
        if isField(layer.c_) :
            c_learning(layer, t)
        
    if isField(layer.d_) : #Update initial hidden states beliefs
        d_learning(layer)
        
    if isField(layer.e_) : # Update agent habits
        layer.e_ = layer.e_ + layer.u[:,T-1]*layer.eta
    
    # # Negative freeee eneergiiiies
    # for modality in range (Nmod):
    #     if isField(layer.a_):
    #         layer.Fa.append(-spm_KL_dir(layer.a_[modality],layer.a_prior[modality]))
    #     if isField(layer.c_) :
    #         layer.Fc.append(- spm_KL_dir(layer.c_[modality],layer.c_prior[modality]))
    
    # for factor in range(Nf):
    #     if isField(layer.b_):
    #         layer.Fb.append(-spm_KL_dir(layer.b_[factor],layer.b_prior[factor]))
    #     if isField(layer.d_):
    #         layer.Fd.append(-spm_KL_dir(layer.d_[factor],layer.d_prior[factor]))
    
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