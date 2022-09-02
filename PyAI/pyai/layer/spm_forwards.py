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

from ..base.function_toolbox import normalize,spm_dot, nat_log,softmax
from ..base.miscellaneous_toolbox import isNone,flatten_last_n_dimensions,flexible_toString,flexible_print,flexible_copy
from ..visi_lib.state_tree import tree_node,state_tree

def spm_forwards(O,P,U,A,B,C,E,A_ambiguity,A_novelty,B_novelty,t,T,N,t0 = 0,verbose = False) :
    """ 
    Recursive structure, each call to this function provides the efe and expected states at t+1


    Environment rules (A,B,C,E,A_ambiguity,A_novelty)                         Overall EFE for next action at time t: G
    States given observations at time t : P(t)                       Predictive weighted distribution of states at time t1 given EFE
        |                                                                                         / \ 
        |                                                                                          |
       \_/                                                                                         |
    Tree of possible states and obs following                                        Resulting EFE for each policy choice
    plausible policies P(up to t+N)             ---------------------------->          is summed from t+N to t : G
                                                (reccurent use of spm_forwards)        EFE is calculated as a risk term (exploit)
                                                                                        and an ambiguity term (explore)
    Note : ambiguity regarding future OBSERVATIONS ? Not hidden states --> not adapted for uncertain A ?
                        Correlated to but not directly linked to hidden state uncertainty
                        THe agent strives to reduce uncertainty regarding the feedback, but
                        does not want to reduce uncertainty regarding current hidden state
                        --> If A is unknown and B is known, there will be no incentive towards current
                        hidden state uncertainty reduction ?

                        "Ambiguous states are those that have an uncertain mapping to
                        observations. The greater these quantities, the less likely
                        it is that the associated policy will be chosen."
                                Generalised free energy and active inference
                                Thomas Parr  · Karl J. Friston

    
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
    # P is the posterior over hidden states at the current time t based on priors
    P[t] =normalize(L.flatten()*P[t])

    if (t==T):
        # Search over, calculations make no sense here as no actions remain to be chosen
        return normalize(np.ones(G.shape)), P
        

    Q = []
    for action in range(U.shape[0]) :
        Q.append(np.dot(B[action],P[t])) # predictive posterior of states at time t
        # print(Q)
        for modality in range(Nmod):
            # print(Nf)
            # print(A[modality].shape)
            flattened_A = flatten_last_n_dimensions(A[modality].ndim-1,A[modality])
            flattened_W = flatten_last_n_dimensions(Nf,A_novelty[modality])
            qo = np.dot(flattened_A,Q[action]) # prediction over observations at time t
            po = C[modality][:,t]              # what we want at that time
            bayesian_risk_only = False 
            if (bayesian_risk_only):
                G[action] = G[action] + np.dot(qo.T,po) # ROI if only bayesian risk is computed
            else :
                ambiguity = np.dot(Q[action].T,A_ambiguity[modality].flatten()) # I'd rather solve uncertainty
                risk =  - np.dot(qo.T,nat_log(qo)-po) # I want to go towards preferable results
                # G[factor] =                              ambiguity              +                 risk
                G[action] = G[action] + ambiguity + risk


                # Bayesian surprise about parameters (= novelty)
                # A term to promote agent exploration to improve the model
                # The smaller the weights of a single cell, the more it is explored
                # The bigger the weights of the whole modality, the less it is explored
                # No sign error, we calculate MINUS EFE and not EFE in this routine
                A_exploration_term = 0
                B_exploration_term = 0

                # if we learn a :
                A_exploration_term = np.dot(qo.T,np.dot(flattened_W,Q[action]))
                # if we learn b : 
                # Q[action] is the posterior over states$

                # We calculate how much new information would be gained by picking action B knowing that we would be 
                # starting from a distribution of state P and going towards a distribution of states Q
                # Q = q(s+1|pi) at time t
                # P = q(s) at time t
                B_exploration_term = np.dot(Q[action],np.dot(B_novelty[action],P[t]))

                debug = False
                if (t==t0) and debug :
                    print(t)
                    print("Action : " + str(action) + "  - Exploratory terms : B " + str(B_exploration_term) + " A " + str(A_exploration_term))
                    print("Compared to G[action] = " + str(G[action]) + "   from :")
                    print("       - Ambiguity = " + str(ambiguity))
                    print("       - Risk      = " + str(risk))
                G[action] = G[action] - A_exploration_term - B_exploration_term
                # print(G,action,A_exploration_term)
                # print(np.round(B_novelty[action],2))
                if (t==t0) and debug :
                    print("Resulting in G[action] = " + str(G[action]))
                    print("---")
                # Let's pick an observation probability dirichlet prior for given set of states : a_1 = [10,5,0.01], a_2  = [1,1,1]
                # w_norm(a_1) = -[0.1 - 1/15, 0.2 - 1/15, 100 - 1/15]) = [-0.033, -0.133 , -99.933]
                # w_norm(a_2) = -[1 - 1/3 ,1 - 1/3 ,1 - 1/3] =           [-0.66,-0.66,-0.66]         # Considerable weights towards a_1, whereas it has already been explored --> error !
                # WOuld encourage very well-known actions & particularly unwanted actions
                # The inverse would make a lot more sense ? (R. Smith et al. , A step-by-step tutorial on active inference and its application to empirical data, 2022 page 34)

                # TODO :  b ambiguity term ?
                
                # print(np.dot(B_novelty[action],P[t]))
                # print(Q[action])
                #print(t,action,A_exploration_term)
                # print("#######")

                # Remark : leave as is for now, but not the best way to 
                # promote A and B exploration ?
    # Q = q(s|pi) at time t
    # P = q(s) at time t
    # P_archive = q(s) at time t --> N 
    # u = q(pi) at time t
    # if (t==0):
    #     print(G)
    #     np.set_printoptions(suppress=True)
    #     print("A_novelty =")
    #     print(np.round(flattened_W,2))
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
    debug = False 
    if debug :
        print("#########################")
        print(np.round(G,2))
        print(softmax(G))
        print("#########################")
    if (t<N): # t within temporal horizon
        u = softmax(G)
        k = (u<plausible_threshold)
        u[k] = 0
        G[k] = -1000
        K = np.zeros((Q[0].shape))
        for action in range(U.shape[0]) :           

            if (u[action]>=plausible_threshold):
                indexes = np.where(Q[action]>plausible_threshold)[0]
                #print(">>>  at time t =  " + str(t) + "   --- action = " + str(action))
                if(indexes.size == 0):
                    indexes = np.where(Q[action]>1/Q[action].size)[0]
                indexes = tuple(indexes)
                for index in indexes :
                    for modality in range (Nmod):
                        O[modality] = flatten_last_n_dimensions(Nf,A[modality])[:,index]

                    #prior over subsequent action under this hidden state
                    #----------------------------------------------------------
                    P[t+1] = Q[action]
                    F,useless = spm_forwards(flexible_copy(O),P,U,A,B,C,E,A_ambiguity,A_novelty,B_novelty,t+1,T,N,t0=t0)
                    
                    if not(isNone(F)):  
                        # If the next timestep is not the last, update efe marginalized over subsequent action
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
    return G,P


# TODO : for T = 3, t = 0, prediction of the last step is the same as P[0] = [1,0,...,0]
# Probably due to backpropagation of P. Wanted behaviour ?