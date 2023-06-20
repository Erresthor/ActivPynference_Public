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

from ...base.miscellaneous_toolbox import isField, flexible_copy
from ...base.function_toolbox import normalize , spm_kron, spm_wnorm, nat_log , spm_psi, softmax
from ...base.function_toolbox import inverted_spm_wnorm
from .parameters.policy_method import Policy_method
from .layer_precisions import init_precisions




def initialize_sizes(layer_input):
    """This function initialize the sizes of all blocks"""
    T = layer_input.T
    layer_input.options.T_horizon = min(T-1,layer_input.options.T_horizon)

    if (layer_input.Nf==0):
        layer_input.Nf = len(layer_input.D_)

    if (layer_input.policy_method== Policy_method.UNDEFINED):
        if (isField(layer_input.U_)):
            layer_input.V_ = np.zeros((T-1,)+layer_input.U_.shape)
            layer_input.V_[0,...] = layer_input.U_
            layer_input.policy_method = Policy_method.ACTION
        elif (isField(layer_input.V_)):
            layer_input.U_ = layer_input.V_[0,:,:]
            layer_input.policy_method = Policy_method.POLICY
        else :
            print("No  U or V as input. Assumig a passive setup")
            layer_input.policy_method = Policy_method.PASSIVE
            layer_input.V_ = np.zeros((T-1,1,layer_input.Nf))

    if(layer_input.policy_method==Policy_method.POLICY):
        layer_input.Np = layer_input.V_.shape[1] # Number of allowable policies
    elif(layer_input.policy_method==Policy_method.ACTION):
        layer_input.Np = layer_input.U_.shape[0] # Number of allowable set of actions

    

    layer_input.Nu = []    # Number of allowable actions for each factor
    for f in range(layer_input.Nf) :
        assert layer_input.B_[f].ndim > 2,"B_ has too little dimensions"
        layer_input.Nu.append(layer_input.B_[f].shape[2])
        # B_[f] shoulc always be a >= 3D matrix

    
    if (layer_input.Nmod==0):
        layer_input.Nmod = len(layer_input.A_)
    if(layer_input.No ==[]):
        layer_input.No = []
        for i in range(layer_input.Nmod) :
            layer_input.No.append(layer_input.A_[i].shape[0])
    
    
    if (layer_input.Ns==[]):
        layer_input.Ns = []
        for i in range(layer_input.Nf):
            layer_input.Ns.append(layer_input.D_[i].shape[0])
    
    layer_input.N_induced_precisions = layer_input.precisions.count_inherited_precisions()

    if (layer_input.options.Ni < 0) :
        layer_input.options.Ni = 1


    if(layer_input.name == '') :
        layer_input.name = 'unnamed_model'



def initialize_fields(layer_input):
    epsilon = 1e-16
    Nmod = layer_input.Nmod
    No = layer_input.No
    Nf = layer_input.Nf
    Ns = layer_input.Ns  
    Np = layer_input.Np
    Nu = layer_input.Nu
    Ni = layer_input.options.Ni
    t = layer_input.t
    T = layer_input.T

    # Use inputs to initialize blocks :
            # Likelihood model a / A
    assert isField(layer_input.A_), "A_ not filled in"
    A = normalize(layer_input.A_)   # <=> MDP(m).A{g}
    if isField(layer_input.a_):
        a = normalize(layer_input.a_)   # <=> A{m,g}
        a_prior = []             # <=> pA{m,g}
        a_novelty = []          # <=> W{m,g}
        for modality in range(Nmod):
            a_prior.append(np.copy(layer_input.a_[modality]))
            # a_novelty.append( spm_wnorm(a_prior[modality],epsilon)*(a_prior[modality]>epsilon) )
            # I believe the true novelty should be the opposite of the previous term to be >0 :
            a_novelty.append(spm_wnorm(a_prior[modality])*(a_prior[modality]>epsilon) )
        # print("--------------------------------")
        # print("--------------------------------")
        # print("--------------------------------")
        # print("--------------------------------")
        # np.set_printoptions(suppress=True)
        # print(np.round(a_prior[modality],10))
        # print(np.round(a_novelty,2))
        # print("--------------------------------")
        # print("--------------------------------")
        # print("--------------------------------")
        # print("--------------------------------")
    elif isField(layer_input.A_) :
        a = normalize(layer_input.A_)

        a_novelty = []          # <=> W{m,g}
        for modality in range(Nmod):
            a_novelty.append(np.zeros(layer_input.A_[modality].shape))
    else :
        raise RuntimeError("- No perception matrix A as input.")
    

    a_ambiguity = []          # <=> H{m,g}
    for modality in range(Nmod):
        a_ambiguity.append(np.sum(a[modality]*nat_log(a[modality]),0))
        # TODO : Check the calculations here

    # Transition model b / B
    assert isField(layer_input.B_), "B_ not filled in"
    B = normalize(layer_input.B_)
    if isField(layer_input.b_): # If we are learning the transition matrices
        b = normalize(layer_input.b_)
        
        b_prior = []
        b_complexity = []
        b_concentration = []
        for factor in range(Nf):   # For all factors
            b_prior.append(np.copy(layer_input.b_[factor]))
            b_complexity.append(spm_wnorm(b_prior[factor])*(b_prior[factor]>epsilon))
    elif isField(layer_input.B_) :
        b = normalize(layer_input.B_)
    else :
        raise RuntimeError("- No transition matrix B as input.")
    
    if (layer_input.policy_method == Policy_method.ACTION):
        # Kronecker form of policies :
        b_kron = [] 
        b_complex_kron = []
        for k in range(Np) :
            b_kron.append(1)
            b_complex_kron.append(1)
            for f in range(Nf):
                #b_kron[k] = spm_kron(b[f][:,:,layer_input.U_[k,f]],b_kron[k])
                b_kron[k] = spm_kron(b_kron[k],b[f][:,:,layer_input.U_[k,f]])
                b_complex_kron[k] = spm_kron(b_complex_kron[k],b_complexity[f][:,:,layer_input.U_[k,f]])                
    # Some way of "compressing" multiple factors into a single matrix 
    # Different from Matlab script, because our kronecker product orders dimension differently


    # prior over initial states d/D
    if isField(layer_input.d_):
        eps = 1e-10
        for f in range(Nf):
            if (np.sum(layer_input.d_[f])<eps) :
                layer_input.d_[f] += eps
        d = normalize(layer_input.d_)
        
        d_prior = []
        d_complexity = []
        for f in range(Nf):
            d_prior.append(np.copy(layer_input.d_[f]))
            d_complexity.append(spm_wnorm(d_prior[f]))
    elif isField(layer_input.D_) :
        d = normalize(layer_input.D_)
    else :
        d = []
        for f in range(Nf):
            d.append(normalize(np.ones(Ns[f],)))
        layer_input.D_ = d
    D = normalize(layer_input.D_)
    
    # Habit E
    if isField(layer_input.e_):
        E = layer_input.e_
        e_prior = np.copy(layer_input.e_)
    elif isField(layer_input.E_) :
        E = layer_input.E_
    else :
        E = np.ones((Np,))
    E = nat_log(E/sum(E))
    
    # Preferences C
    C = []
    if isField(layer_input.c_):
        c_prior = []
        for modality in range(Nmod):
            C.append(spm_psi(layer_input.c_[modality] + 1./32))
            c_prior.append(np.copy(layer_input.c_[modality]))      
    elif isField(layer_input.C_):
        for modality in range(Nmod):
            C.append(layer_input.C_[modality])
    else : 
        for modality in range(Nmod):
            C.append(np.zeros((No[modality],1)))
    
    for modality in range(Nmod):
        assert(C[modality].ndim>1),"C should be at least a 2 dimensional matrix. If preferences are time-invariant, the second dimension should be of size 1."
        if (C[modality].shape[1] == 1) :
            C[modality] = np.tile(C[modality],(1,T))
            if (layer_input.c_):
                layer_input.c_[modality] = np.tile(layer_input.c_[modality],(1,T))
                c_prior[modality] = np.tile(c_prior[modality],(1,T))
        C[modality] = nat_log(softmax(C[modality],0))
    
    if (layer_input.policy_method == Policy_method.POLICY) :
        V = layer_input.V_.astype(np.int)
        layer_input.V = V
    elif (layer_input.policy_method == Policy_method.ACTION) : 
        U = layer_input.U_.astype(np.int)
        layer_input.U = U
    else :
        assert (Np == 1), "If V_ is not input, there should only be one policy available .."
        V = np.zeros((T,1,Nf))
        layer_input.V = V

    # --------------------------------------------------------
    layer_input.a = a
    layer_input.A = A
    if isField(layer_input.a_):
        layer_input.a_prior = a_prior
    layer_input.a_complexity = a_novelty
    layer_input.a_ambiguity = a_ambiguity

    layer_input.b = b
    layer_input.B = B
    if isField(layer_input.b_):
        layer_input.b_prior = b_prior
        layer_input.b_complexity = b_complexity
    if (layer_input.policy_method == Policy_method.ACTION):
        layer_input.b_kron = b_kron
        layer_input.b_complexity = b_complex_kron
    
    layer_input.c = C
    layer_input.C = C
    if isField(layer_input.c_):
        layer_input.c_prior = c_prior
    
    layer_input.d = d
    layer_input.D = D
    if isField(layer_input.d_):
        layer_input.d_prior = d_prior
        layer_input.d_complexity = d_complexity
    
    layer_input.e = E
    # --------------------------------------------------------

    # the value inside observations/ true states  can be:
    # <0 if not yet defined : it will be created by the generative process
    # x in [0,No[modality]-1]/[0,Ns[factor]-1] for a given modality/factor at a given time if we want to explicit some value

    #OUTCOMES -------------------------------------------------------------
    layer_input.O = []
    for modality in range(Nmod):
        layer_input.O.append(np.zeros((No[modality],T)))

    o = np.full((Nmod,T),-1)      
    if isField(layer_input.o_):  # There is an "o" matrix with fixed values ?
        if (type(layer_input.o_) == np.ndarray) :
            o[layer_input.o_ >=0] = layer_input.o_[layer_input.o_>=0]
            
        # If there are fixed values for the observations, they will be copied
    layer_input.all_outcomes_input = (np.sum(o<0)==0)  # If we have all outcomes as input, no need to infer them 
    layer_input.o = np.copy(o)

    #STATES ------------------------------------------------------------------
    # true states 
    s = np.full((Nf,T),-1)
    if isField(layer_input.s_):
        if (type(layer_input.s_) == np.ndarray) :
            s[layer_input.s_ >=0] = layer_input.s_[layer_input.s_ >=0]
    layer_input.all_states_input = (np.sum(s<0)==0)  # If we have all states as input, no need to infer them 
    layer_input.s = np.copy(s)
    
    # Posterior expectations of hidden states
    # state posteriors
    layer_input.x = []
    layer_input.xn = []
    layer_input.X = []
    layer_input.X_archive = []
    layer_input.S = []
    for f in range(Nf):
        layer_input.x.append(np.zeros((Ns[f],T,Np)) + 1./Ns[f])                     # Posterior expectation of all hidden states depending on each policy
        layer_input.xn.append(np.zeros((Ni,Ns[f],T,T,Np)) + 1./Ns[f])
        layer_input.X.append(np.tile(np.reshape(d[f],(-1,1)),(1,T)))                # Posterior expectation of all hidden states at the current time
        layer_input.X_archive.append(np.tile(np.reshape(d[f],(-1,1,1)),(T,T)))      # Estimation at time t of BMA states at time tau
        layer_input.S.append(np.zeros((Ns[f],T,T)) + 1./Ns[f])                      # Posterior expectation of all hidden states over time
        for k in range(Np):
            layer_input.x[f][:,0,k] = d[f]

    layer_input.Q = []
    for t in range(T):
        layer_input.Q.append(spm_kron(D))


    layer_input.vn = []
    for f in range(Nf):        
        layer_input.vn.append(np.zeros((Ni,Ns[f],T,T,Np)))  # Recorded neuronal prediction error
    
    #ACTIONS ------------------------------------------------------------------
    # >>> Action_selection

    #history of posterior over action
    layer_input.u_posterior_n = np.zeros((Np,Ni*T))             
    #posterior over action
    layer_input.u_posterior = np.zeros((Np,T))                
    
    # >>> Chosen Action
    u_temp = np.full((Nf,T-1),-1)  
    if isField(layer_input.u_):
        u_temp[layer_input.u_>=0] = layer_input.u_[layer_input.u_>=0]
    layer_input.u = u_temp


    layer_input.K = np.full((T-1,),-1)
    #POLICIES ------------------------------------------------------------------
    #history of posterior over policy
    layer_input.p_posterior_n = np.zeros((Np,Ni*T))             
    #posterior over policy
    layer_input.p_posterior = np.zeros((Np,T))                 
    if (Np == 1) :
        layer_input.p_posterior = np.ones((Np,T))
            
    # Allowable policies
    p = np.zeros((Np,))
    for policy in range(Np): # Indices of allowable policies
        p[policy] = policy
    layer_input.p = p.astype(np.int)


    #Posterior over policies & actions
    layer_input.P = np.zeros((tuple(Nu)+(1,)))


    # -------------------------------------------------------------
    #TODO : Initialize output variables for a run
    layer_input.L = []


    layer_input.F = np.zeros((layer_input.Np,layer_input.T))
    layer_input.G = np.zeros((layer_input.Np,layer_input.T))
    layer_input.H = np.zeros((layer_input.T,))     
    



    layer_input.w = np.zeros((T,))  # Policy precision w
    layer_input.wn = None          # Neuronal encoding of policy precision

    layer_input.dn = None          # Simulated dopamine response
    layer_input.rt = np.zeros((T,))          # Simulated reaction times
    
    

    init_precisions(layer_input)

def prep_layer(input_layer):
    initialize_sizes(input_layer)
    initialize_fields(input_layer)