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
from .model_avg_utils import joint_to_kronecker_accross_time,kronecker_to_joint_accross_time#,kronecker_to_joint,joint_to_kronecker

from ..enums.memory_decay import MemoryDecayType
from ..enums.space_structure import AssumedSpaceStructure

# This whole page should be updated using the einsums ! 
# --> use jax_functions here ?

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
    if (structure_assumption == AssumedSpaceStructure.NO_STRUCTURE):
        return base_information
    
    # If we assume a linear structure within this factor's state space
    if ("LINEAR" in structure_assumption.name) :
        clamp_interp = (structure_assumption==AssumedSpaceStructure.LINEAR_CLAMPED)
        periodic_interp = (structure_assumption==AssumedSpaceStructure.LINEAR_PERIODIC)
        generalized_information = extrap_diag_2d(base_information,clamp_interp,periodic_interp,
                                                 fadeout_function,False)
    return generalized_information

def a_learning(o_d_history,s_kron_d_history,old_a_matrix,
               plasticityOptions):
    """
    TODO : reimplement with einsums !
    Returns an updated perception matrix given an history of :
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
    """
    TODO : reimplement with einsums !
    Returns an updated transition matrix given an history of :
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

    L = spm_backwards(o_margin_history, x_kron_history, a_kron, kronecker_transition_history)

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
        d_estimate = s_margin_history[factor][:,0][i]
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
    Ns = layer.Ns

    # Layer plasticity : Update all learning options ---------------------------
    # TODO : this may be directly done in the learning parameters ?
    backwards_pass = layer.learn_options.backwards_pass
    eta = layer.learn_options.eta
    mem_loss = layer.learn_options.memory_loss
    mem_decay_type = layer.learn_options.decay_type
    
    assume_state_space_structure = layer.learn_options.assume_state_space_structure
    fadeout_function = layer.learn_options.get_generalize_fadeout_function()
    
    if (type(assume_state_space_structure)==list):
        if not(len(assume_state_space_structure)==Nf):
            assert len(assume_state_space_structure)==1, "Your defined structural assumptions list length does not match the number of states : " + str(len(assume_state_space_structure)) + " =/= " + str(Nf)
            
            assume_state_space_structure = assume_state_space_structure[0]
            print("WARNING : Got one value in the list of space structure assumption. Assuming that this assumption is valid for all states factors.")
            print("If this is wanted behviour and you want to hide this message, please set your assumed state space structure to AssumedSpaceStructure type.")
    general_plasticity = layerPlasticity(eta,mem_decay_type,mem_loss,assume_state_space_structure,fadeout_function)

    
    
    # Last trial's history :
    STM = layer.STM

    o_history = STM.o
    o_d_history = STM.o_d
    
    x_history = STM.x
    x_d_history = STM.x_d
    x_kron_history = joint_to_kronecker_accross_time(x_d_history)

    u_history = STM.u
    u_d_history = STM.u_d
    
    # LEARNING :
    marginalized_o = spm_complete_margin(o_d_history,o_d_history.ndim-1)
    marginalized_x = spm_complete_margin(x_d_history,x_d_history.ndim-1)

    # Get an history of performed state transitions
    using_realized_actions = True  # If there is ambiguity regarding the last selected action
                                   # Useful in contexts such as BCI ?
    if using_realized_actions:
        transition_history_u_d = dist_from_definite_outcome_accross_t(u_history,u_d_history.shape)
            # One_hot encoding of the realized action
    else :
        transition_history_u_d = u_d_history
            # Using the posterior distribution only (when the subject is not sure of the performed action)
    b_kron_array = np.stack(layer.var.b_kron,axis=-1) # Shape Ns x Ns x Np
    history_of_state_transitions = np.einsum("iju,ut->tij",b_kron_array,transition_history_u_d)
    
    
    # Perform a HMM backward pass to "smooth the probabilities"
    if backwards_pass :
        smoothed_x_kron_history = backward_state_posterior_estimation(marginalized_o,x_kron_history,layer.var.a_kron,history_of_state_transitions)
        STM.x_d_smoothed = kronecker_to_joint_accross_time(smoothed_x_kron_history,layer.Ns) # Let's save it to the layer's STM !
        marginalized_smoothed_x = spm_complete_margin(STM.x_d_smoothed,x_d_history.ndim-1)
    else :
        smoothed_x_kron_history = x_kron_history
        marginalized_smoothed_x = marginalized_x
        # Nothing to save to the STM, because states were not smoothed


    if (layer.learn_options.learn_a): 
        new_a = a_learning(marginalized_o,smoothed_x_kron_history,layer.a,
                general_plasticity)
        layer.a = new_a
    
    if (layer.learn_options.learn_b):
        new_b = b_learning(transition_history_u_d,marginalized_smoothed_x,layer.b,layer.U,
               general_plasticity) 
        layer.b = new_b

    
    
    if (layer.learn_options.learn_d) : #Update initial hidden states beliefs
        if (layer.learn_options.use_backward_pass_to_learn_d):
            new_d = d_learning_smooth(marginalized_smoothed_x,layer.d,general_plasticity)
        else :
            new_d = d_learning_base(marginalized_o,x_kron_history,layer.var.a_kron,
                            history_of_state_transitions,layer.d,
                    general_plasticity)
        layer.d = new_d   
        
    
    if (layer.learn_options.learn_c) :
        c_learning(marginalized_o,layer.c,general_plasticity)
        
    if (layer.learn_options.learn_e): # Update agent habits
        e_learning(transition_history_u_d,layer.e,general_plasticity)
    
    # For now, we stop here, but more indicators could be computed here