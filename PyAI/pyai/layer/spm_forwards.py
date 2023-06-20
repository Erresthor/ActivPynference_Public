# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:44:10 2021

@author: Côme ANNICCHIARICO(come.annicchiarico@inserm.fr), adaptation of the work of :

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
from ..base.miscellaneous_toolbox import isNone,flexible_copy

def spm_forwards(O,P_t,U,layer_variables,t,T,N,policy_tree_node,
                 MAX_ACTION_BRANCHES_PER_TIMESTEP=3,MAX_STATE_BRANCHES_PER_ACTION=3) :
    """ 
    Recursive structure, each call to this function provides the efe and expected states at t+1


    Environment rules (A,B,C,E,A_ambiguity,A_novelty)                         Overall EFE for next action at time t: G
    States given observations at time t : P(t)                       Predictive weighted distribution of states at time t1 given EFE
        |                                                                                         / \ 
        |                                                                                          |
       \_/                                                                                         |
    Tree of possible states and obs following                                        Resulting EFE for each policy choice
    plausible policies P(up to t+N)             ---------------------------->          is summed from t+N to t : G
                                                (recurrent use of spm_forwards)        EFE is calculated as a risk term (exploit)
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
    B = self.b
    """
    # print("----->")
    # print(policy_tree_node.deep_index)
    # print(policy_tree_node.deep_index)
    A = layer_variables.a
    flatten_a = layer_variables.a_kron
    flatten_a_novelty = layer_variables.a_kron_novelty
    A_ambiguity = layer_variables.a_ambiguity

    B = layer_variables.b_kron
    C = layer_variables.c
    E = layer_variables.e
    
    B_novelty = layer_variables.b_kron_complexity

    Nf = A[0].ndim-1  # granted A has at least 1 modality, any situation without observation isn't explored here
    Nmod = len(A)
    P = flexible_copy(P_t)
    G = flexible_copy(E)

    # L is the posterior over hidden states based on likelihood (A & O)
    L = 1
    for modality in range (Nmod):
        L = L * spm_dot(A[modality],O[modality]) 
    # P is the posterior over hidden states at the current time t based on priors
    # print(G)
    P =normalize(L.flatten()*P) # P(s|o) = P(o|s)P(s)
    policy_tree_node.update_state_posterior(P)

    if (t==T):
        # Search over, calculations make no sense here as no actions remain to be chosen
        # Return predicted state at time T (todo : CHECK T-1)
        return normalize(np.ones(G.shape)), P
        
    Q = [] # Q is the predictive posterior of states at the current time t depending on the selected action
            # Q = q(s|pi)
    # For now, G is the prior over policies p(pi)
    # The following loop will compute an approximation of the free energy of the future (?)
    # To approximate the posterior over policie. To do that, it balances out states epistemics and 
    # eploitative elements (risk), as well as model exploration terms.
    for action in range(U.shape[0]) :
        Q.append(np.dot(B[action],P)) 
        for modality in range(Nmod):
            # flattened_A = flatten_last_n_dimensions(A[modality].ndim-1,A[modality])
            # flattened_W = flatten_last_n_dimensions(Nf,A_novelty[modality])

            flattened_A = flatten_a[modality]
            flattened_W = flatten_a_novelty[modality]
            qo = np.dot(flattened_A,Q[action]) # prediction over observations at time t
            po = C[modality][:,t]              # what we want at that time , log(p(o))

            bayesian_risk_only = False 
            if (bayesian_risk_only):
                G[action] = G[action] + np.dot(qo.T,po) # ROI if only bayesian risk is computed
            else :
                ambiguity = np.dot(Q[action].T,A_ambiguity[modality].flatten()) # I'd rather solve uncertainty
                risk =  - np.dot(qo.T,nat_log(qo)-po) # I want to go towards preferable results = - D_KL[q(o|pi)||p(o)]
                # G[factor] =                              ambiguity              +                 risk
                G[action] = G[action] + ambiguity + risk

                # Exploration terms
                A_exploration_term = 0
                B_exploration_term = 0

                # if we learn a :
                A_exploration_term = np.dot(qo.T,np.dot(flattened_W,Q[action]))

                # if we learn b :
                B_exploration_term = np.dot(Q[action],np.dot(B_novelty[action],P))
                # if(t==0):
                #     print("EXPLOR : "+str(action))
                #     print(A_exploration_term)
                #     print(B_exploration_term)
                G[action] = G[action] - A_exploration_term - B_exploration_term
    
    # Q = q(s|pi) at time t
    # P = q(s) at time t
    # u = softmax(G) = q(pi) at time t

    # Not over yet, G still needs to include the "value" of the next timesteps !
    # Let's imagine action k is realized :
    # 1.  Let's check if it is actually plausible or not ? If not, why even bother
    # 2.  LEt's then check the plausible states for this transition (Q[k])
    #       2.a : are they plausible ? If not, just skip it
    #       2.b : if yes, then what are the associated observation distribution according to my model ?
    # 3. Using those computed values, if all plausible, let's approximate the

    plausible_threshold = 1.0/16.0
    if (t<N): # t within temporal horizon
        u = softmax(G)
        policy_tree_node.update_policy_prior(u)

        mask_action_not_explored = (u<plausible_threshold)
        u[mask_action_not_explored] = 0
        sorted_u = np.argsort(-u)
        # print(u,sorted_u)
        G[mask_action_not_explored] = -1000
        K = np.zeros((Q[0].shape))
        for action in range(U.shape[0]) :           
            
            if (u[action]>=plausible_threshold): #If this action is plausible
                indexes = np.where(Q[action]>plausible_threshold)[0]
                
                if(indexes.size == 0):
                    indexes = np.where(Q[action]>1/Q[action].size)[0]
                indexes = tuple(indexes) # All the indices of plausible next states for this action
                # print(indexes)
                for index in indexes :
                    # print(index)
                    for modality in range (Nmod):
                        # O[modality] = flatten_last_n_dimensions(Nf,A[modality])[:,index]
                        O[modality] = flatten_a[modality][:,index]

                    #prior over subsequent action under this hidden state
                    #----------------------------------------------------------
                    P_next_t = Q[action]

                    child_node = policy_tree_node.add_child(P_next_t,u_index_in = action,s_index_in=index)
                    F,posterior_P_next_t = spm_forwards(flexible_copy(O),P_next_t,U,layer_variables,t+1,T,N,child_node)
                    
                    if not(isNone(F)):  
                        # If the next timestep is not the last, update efe marginalized over subsequent action
                        # Structure could be optimized, but will do for now
                        # Expected free energy marginalised over subsequent action
                        K[index] = np.inner(softmax(F),F)
                indexes_L = list(indexes)
                G[action] = G[action] + np.dot(K[indexes_L],Q[action][indexes_L])
        u = softmax(G)
        R = 0

        # Posterior over next state marginalised over subsequent action
        for action in range(U.shape[0]):
            R = R + u[action]*Q[action]
        pol_w_next_s_posterior = R

        policy_tree_node.update_policy_posterior(u)
        policy_tree_node.update_pol_weighted_next_state_posterior(pol_w_next_s_posterior)
    return G,P


# TODO : for T = 3, t = 0, prediction of the last step is the same as P[0] = [1,0,...,0]
# Probably due to backpropagation of P. Wanted behaviour ?