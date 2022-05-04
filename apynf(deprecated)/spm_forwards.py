# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:44:10 2021

@author: Côme ANNICCHIARICO(come.annicchiarico@mines-paristech.fr), adaptation of the work of :

implementation of 
%__________________________________________________________________________
% Copyright (C) 2019 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_MDP_VB_XX.m 7766 2020-01-05 21:37:39Z karl $
(SPM12's "spm_forwards' function from MDP_VB_XX)
"""



"""% deep tree search over policies or paths
%--------------------------------------------------------------------------
% FORMAT [G,Q] = spm_G(O,P,A,B,C,E,H,W,t,T);
% O{g}   - cell array of outcome probabilities for modality g
% P      - empirical prior over (vectorised) states
% A{g}   - likelihood mappings from hidden states
% B{k}   - belief propagators (action dependent probability transitions)
% C{g}   - cost: log priors over future outcomes)
% E      - empirical prior over actions
% H{g}   - state dependent ambiguity
% W{g}   - state and outcome dependent novelty
% t      - current time point
% T      - time horizon
% N      - policy horizon
%
% Q      - posterior over (vectorised) states for t
%
%  This subroutine performs a deep tree search over sequences of actions to
%  evaluate the expected free energy over policies or paths. Crucially, it
%  only searches likely policies under likely hidden states in the future.
%  This search is sophisticated; in the sense that posterior beliefs are
%  updated on the basis of future outcomes to evaluate the free energy
%  under each outcome. The resulting  average is then accumulated to
%  furnish a path integral of expected free energy for the next action.
%  This routine operates recursively by updating predictive posteriors over
%  hidden states and their most likely outcomes.
%__________________________________________________________________________
This SPM12"s function has been modified to include an archive of expectations 
of future states using a tree. 
"""

import numpy as np
from types import SimpleNamespace

from base.function_toolbox import normalize,spm_dot, nat_log,softmax
from base.miscellaneous_toolbox import isNone,flatten_last_n_dimensions,flexible_toString,flexible_print,flexible_copy
from base.state_tree import tree_node,state_tree

def spm_forwards(O,P,U,A,B,C,E,A_ambiguity,A_novelty,t,T,N,current_node,t0 = 0,verbose = False) :
    """ 
    Recursive structure, each call to this function provides the efe and expected states at t+1

    States given observations P(t)                         Overall EFE for next action at time t: G
        |                                                                  / \ 
        |                                                                   |
       \_/                                                                  |
    Tree of possible states following                       Resulting EFE for each policy choice
    plausible policies P(up to t+N)      --------------->          is summed from t+N to t : G

    O = O
    P = Q
    A = self.a
    B = self.b"""
    
    
    # Nf = len(B) --> not because B is in Kronecker form
    Nf = A[0].ndim-1  # granted A has at least 1 modality, but any situation without observation isn't explored here
    Nmod = len(A)
    P = np.copy(P)
    G = np.copy(E)
    efe = np.zeros((U.shape))
    # L is the posterior over hidden states based on likelihood (A & O)
    L = 1
    for modality in range (Nmod):
        L = L * spm_dot(A[modality],O[modality])
    # P is the posterior over hidden states based on priors
    P[t] =normalize(L.flatten()*P[t])
    current_node.states = np.copy(P[t])

    if (t==T-1):
        # Search over, calculations make no sense here as no actions remain to be chosen
        return normalize(np.ones(G.shape)), P
        

    Q = []
    for action in range(U.shape[0]) :
        Q.append(np.dot(B[action],P[t])) # predictive posterior of states at time t
        
        for modality in range(Nmod):
            # print(Nf)
            # print(A[modality].shape)
            flattened_A = flatten_last_n_dimensions(A[modality].ndim-1,A[modality])
            qo = np.dot(flattened_A,Q[action]) # prediction over observations at time t
            po = C[modality][:,t]              # what we want at that time
            bayesian_risk_only = False 
            if (bayesian_risk_only):
                G[action] = G[action] + np.dot(qo.T,po) # ROI if only bayesian risk is computed
            else :
                # G[factor] =                              ambiguity              +                 risk
                G[action] = G[action] + np.dot(Q[action].T,A_ambiguity[modality].flatten()) - np.dot(qo.T,nat_log(qo)-po)
                # Bayesian surprise about parameters (= novelty)
                G[action] = G[action] - np.dot(qo.T,np.dot(flatten_last_n_dimensions(Nf,A_novelty[modality]),Q[action]))
    # Q = q(s|pi) at time t
    # P = q(s) at time t
    # P_archive = q(s) at time t --> N 
    # u = q(pi) at time t

    condition = (t<10) and (verbose) and True
    if (t==t0)and (condition):
        print("---------------------------------------------------------------------")
        print("t = " + str(t) + "  || t0 = " + str(t0))
        print("P = " + flexible_toString(P[t]))
        print("Q = \n" + flexible_toString(Q))
        print()
        print("A_novelty = \n" + flexible_toString(A_novelty[0]))
        print("A_amiguity = \n" + flexible_toString(A_ambiguity[0]))
        print("G : " + flexible_toString(G))
        for action in range(U.shape[0]):
            print("    ##### " + str(action)+" #####")
            # If things were simple :
            print("      bayesian_risk[" + str(action) + "] = " + flexible_toString(np.dot(qo.T,po)))
            print("      ---")
            print("      ambiguity[" + str(action) + "]     = " + flexible_toString(np.dot(Q[action].T,A_ambiguity[modality].flatten())))
            print("      risk[" + str(action) + "]          = " + flexible_toString(np.dot(qo.T,nat_log(qo)-po)))
            print("      novelty[" + str(action) + "]       = " + flexible_toString(np.dot(qo.T,np.dot(flatten_last_n_dimensions(Nf,A_novelty[modality]),Q[action]))))
        # print("Ambiguity :")
        # for action in range(U.shape[0]):
        #     print("    G[" + str(action) + "] = " + str(G[action]))
        # print("Risk")

    plausible_threshold = 1.0/16.0
    possibilities_depending_on_action = []
    if (t<N): # t within temporal horizon
        u = softmax(G)
        k = (u<plausible_threshold)
        u[k] = 0
        G[k] = -1000
        K = np.zeros((Q[0].shape))
        for action in range(U.shape[0]) :
            possibilities_depending_on_action.append(0)
            

            if (u[action]>=plausible_threshold):
                indexes = np.where(Q[action]>plausible_threshold)[0]
                #print(">>>  at time t =  " + str(t) + "   --- action = " + str(action))
                if(indexes.size == 0):
                    indexes = np.where(Q[action]>1/Q[action].size)[0]
                indexes = tuple(indexes)
                for index in indexes :
                    # Initialize a node to follow this exploratory path
                    child_node = tree_node(False,t)
                    child_node.action_density = 1.0 # In case we don't update this value
                    child_node.data = action
                    current_node.add_child_node(child_node)

                    #print("              index : " + str(index) + "-----------------")
                    #print("                          "+str(possibilities_depending_on_action))
                    for modality in range (Nmod):
                        O[modality] = flatten_last_n_dimensions(Nf,A[modality])[:,index]

                    #prior over subsequent action under this hidden state
                    #----------------------------------------------------------
                    P[t+1] = Q[action]
                    F,useless = spm_forwards(flexible_copy(O),P,U,A,B,C,E,A_ambiguity,A_novelty,t+1,T,N,child_node,t0=t0)

                    possibilities_depending_on_action[action] += 1
                    
                    if not(isNone(F)):  
                                    # If the next timestep is not the last, update efe marginalised over subsequent action
                                    # Structure could be optimized, but will do for now
                        K[index] = np.inner(softmax(F),F)
                    #print("                           --->")
                    #print("                          "+str(possibilities_depending_on_action))
                indexes_L = list(indexes)
                G[action] = G[action] + np.dot(K[indexes_L],Q[action][indexes_L])

    u = softmax(G)
    R = 0
    for action in range(U.shape[0]):
        R = R + u[action]*Q[action]
    P[t+1] = R

    if (t<N) and (len(current_node.children_nodes)>0):
        counter = 0
        for action in range(U.shape[0]) :
            
            number_of_futures = possibilities_depending_on_action[action]
            for branch in range(number_of_futures):
                current_node.children_nodes[counter].action_density = u[action]*Q[action][Q[action]>plausible_threshold][branch] # One action can lead to several "branches" representing resulting states with various weight. 
                                                                                                                                # To calculate those weight, we use Q[action]. Because we didn't normalize Q[Q>thresh], it could lead to
                                                                                                                                # Problems with the assert statement below, but it'll do for now
                current_node.children_nodes[counter].data = action                                                                                                        
                counter += 1
        somme = 0
        for childnode in current_node.children_nodes :
            somme += childnode.action_density
        assert (somme-1)<1e-6, "Sum of conditionnal probabilities in children nodes should equal to 1. Actual value : " + str(somme)
    
    return G,P


# TODO : for T = 3, t = 0, prediction of the last step is the same as P[0] = [1,0,...,0]
# Probably due to backpropagation of P. Wanted behaviour ?