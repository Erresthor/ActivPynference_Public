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

from ..base.function_toolbox import normalize,spm_dot, nat_log,softmax, prune_tree_auto
from ..base.miscellaneous_toolbox import isField,flexible_copy

def spm_forwards(O,P_t,U,layer_variables,t,
                 T,N,layer_options=None,
                layer_learn_options = None,
                layer_RNG=None,
                depth=0) :
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
                                                                                        and an ambiguity term (explore) (+novelty)

    Note : ambiguity regarding future OBSERVATIONS ? Not hidden states --> not adapted for uncertain A ?
                        "Ambiguous states are those that have an uncertain mapping to
                        observations. The greater these quantities, the less likely
                        it is that the associated policy will be chosen."
                                Generalised free energy and active inference
                                Thomas Parr  · Karl J. Friston

    Note 2 : Each timestep this tree search may explore up to :
        (Number of possible actions x Number of possible states)^(temporal horizon) 
                steps. This may be a lot (if 5 actions & 10 states on 4 Th,  thats 6.25million tree branches).
                Clever solutions probably exist to solve this but for now one can use a simple cap that limits how many
                branches are explored. For now, let's sample randomly from the expected distribution OR sample through how likely a given policy / state is
                OR sample from a distribution with a temperature parameter                 

    O = O
    P = Q
    A = self.a
    B = self.b
    """
    cap_state_explo = layer_options.cap_state_explo
    cap_action_explo = layer_options.cap_action_explo
    compute_novelty_b = layer_options.b_novelty
    
    _depth = depth 


    # print(t)
    # cap_state_explo = 3
    # cap_action_explo = None
    # print(cap_action_explo)
    DETERMINISTIC_PRUNING = True

    # learn_a = True
    # learn_b = True
    # if (isField(layer_learn_options)):

    A = layer_variables.a
    flatten_a = layer_variables.a_kron
    flatten_a_novelty = layer_variables.a_kron_novelty
    A_ambiguity = layer_variables.a_ambiguity

    B = layer_variables.b_kron
    B_novelty = layer_variables.b_kron_complexity

    C = layer_variables.c
    E = layer_variables.e
    
    Nf = A[0].ndim-1  # granted A has at least 1 modality, any situation without observation isn't explored here
    Nmod = len(A)
    P = flexible_copy(P_t)

    G = np.zeros((E.shape[0],6)) # [Nactions]x[6 = prior + risk + ambiguity + (novelty_a + novelty_b) + subsequent_action(s)]
    G[:,0] = E

    # L is the posterior over hidden states based on likelihood (A & O)
    L = 1
    for modality in range (Nmod):
        L = L * spm_dot(A[modality],O[modality]) 
    post_unnormalized = L.flatten()*P # posterior unnormalised
    F = nat_log(np.sum(post_unnormalized))

    # P is the posterior over hidden states at the current time t based on priors
    P =normalize(post_unnormalized) # P(s|o) = P(o|s)P(s)

    if (t==T-1):
        # Search over, calculations make no sense here as no actions remain to be chosen
        # Return predicted state at time T (todo : CHECK T-1)
        return G,P
        return normalize(np.ones(G.shape)), P
        
    Q = [] # Q is the predictive posterior of states at the next time t+1 depending on the selected action
            # Q = q(s|pi)
    # For now, G is the prior over policies p(pi)
    # The following loop will compute an approximation of the free energy of the future (?)
    # To approximate the posterior over policie. To do that, it balances out states epistemics and 
    # eploitative elements (risk), as well as model exploration terms.
    
    for action in range(U.shape[0]) :
        Q.append(np.dot(B[action],P)) 
        for modality in range(Nmod):
            flattened_A = flatten_a[modality]
            flattened_W = flatten_a_novelty[modality]
            flattened_H = A_ambiguity[modality]

            qo = np.dot(flattened_A,Q[action])   # prediction over observations at time t+1

            # plot_action_selection = False
            # if plot_action_selection and (_depth==0) and (action==2):
            #     print("qo ---------------------------")
            #     print(np.round(qo,2))
            #     print("qs ---------------------------")
            #     print(np.round(Q[action],2))
            #     print("novelty")
            #     print(np.round(np.dot(flattened_W,Q[action]),2))
            #     print(np.round(qo*np.dot(flattened_W,Q[action]),2))
            #     print("total")
            #     print(np.round(-np.dot(qo.T,np.dot(flattened_W,Q[action])),2))
            
            po = C[modality][:,t+1]              # what we want at that time (t+1) = log(p(o))

            bayesian_risk_only = False 
            if (bayesian_risk_only):
                risk = np.dot(qo.T,po)
                G[action,1] = G[action,1] + risk # ROI if only bayesian risk is computed
            else :
                risk =  - np.dot(qo.T,nat_log(qo)-po) # I want to go towards preferable results = - D_KL[q(o|pi)||p(o)]
                G[action,1] = G[action,1] + risk

                ambiguity = np.dot(Q[action].T,flattened_H) # I'd rather solve uncertainty and avoid ambiguous observations (i want to minimize the entropy of the observation distribution)
                G[action,2] = G[action,2] + ambiguity

                # Adding exploration terms (reduce uncertainty relative to environment dynamics)
                A_exploration_term = 0
                if layer_learn_options.learn_a : # if we learn a :
                    A_exploration_term = - np.dot(qo.T,np.dot(flattened_W,Q[action]))
                G[action,3] = G[action,3] + A_exploration_term
                
                B_exploration_term = 0
                if compute_novelty_b:
                    if layer_learn_options.learn_b :# if we learn b :
                        # Warning : the novelty term is poorly defined when using several state factors
                        # (dirichlet kronecker product loses part of the information)
                        B_exploration_term = - np.dot(Q[action],np.dot(B_novelty[action],P))
                G[action,4] = G[action,4] + B_exploration_term
    # Q = q(s|pi) at time t
    # P = q(s) at time t
    # u = softmax(G) = q(pi) at time t

    # It's not over yet, G still needs to include the "value" of the next timesteps !
    # Let's imagine action k is realized :
    # 1.  Let's check if it is actually plausible or not ? If not, why even bother
    # 2.  Let's then check the plausible states for this transition (Q[k])
    #       2.a : are they plausible ? If not, just skip it
    #       2.b : if yes, then what are the associated observation distribution according to my model ?
    # 3. Using those computed values, if all plausible, let's approximate the free energy for the analyzed action
    # print("t : " + str(t) + " | " + str(cap_action_explo) + " - " + str(cap_state_explo))

    plausible_threshold = 1.0/16.0
    if (t<N): # t within temporal horizon
        u = softmax(np.sum(G,axis=1))

        if (isField(cap_action_explo)):
            idx_action_to_explore = prune_tree_auto(u,cap_action_explo,DETERMINISTIC_PRUNING,layer_RNG,plausible_threshold=plausible_threshold,
                                                    deterministic_shuffle_between_equal_vals = True)
            idx_action_to_explore = [i[0] for i in idx_action_to_explore] # Convert tuple to int
            mask_action_not_explored = [not(i in idx_action_to_explore) for i in range(u.shape[0])]
        else :
            mask_action_not_explored = (u<plausible_threshold)
        
        u[mask_action_not_explored] = 0
        G[:,5][mask_action_not_explored] = -1000

        # G[tuple(mask_action_not_explored)+(5,)] = -1000
        # print()
        idx_action_to_explore = range(U.shape[0])
        for action in idx_action_to_explore :          
            if (u[action]>=plausible_threshold): #If this action is plausible
                dist_state_to_explore = Q[action]

                if (isField(cap_state_explo)):
                    idx_state_to_explore = prune_tree_auto(dist_state_to_explore,cap_state_explo,DETERMINISTIC_PRUNING,plausible_threshold=plausible_threshold,
                                                           deterministic_shuffle_between_equal_vals = True)
                    idx_state_to_explore = np.array([i[0] for i in idx_state_to_explore]) # Convert tuple to int
                    if(idx_state_to_explore.size == 0):
                        idx_state_to_explore = prune_tree_auto(dist_state_to_explore,cap_state_explo,DETERMINISTIC_PRUNING,plausible_threshold=(1.0/Q[action].shape[0]),
                                                                deterministic_shuffle_between_equal_vals = True)
                        idx_state_to_explore = np.array([i[0] for i in idx_state_to_explore]) # Convert tuple to int
                else :
                    idx_state_to_explore = np.where(dist_state_to_explore>plausible_threshold)[0] 
                    if(idx_state_to_explore.size == 0):
                        plausible_threshold = (1.0/Q[action].shape[0]) # dynamic threshold !
                        idx_state_to_explore = np.where(dist_state_to_explore>plausible_threshold)[0] 

                K = np.zeros((Q[0].shape))
                for index in idx_state_to_explore :
                    for modality in range (Nmod):
                        O[modality] = flatten_a[modality][:,index]

                    #prior over subsequent action under this hidden state
                    #-------------------------------""---------------------------
                    P_next_t = Q[action]
                    

                    G_next_act,posterior_P_next_t = spm_forwards(flexible_copy(O),P_next_t,U,layer_variables,t+1,T,N,
                                                         layer_options=layer_options,
                                                         layer_learn_options=layer_learn_options,
                                                         layer_RNG=layer_RNG,
                                                         depth = _depth+1)
                    
                    # Expected free energy marginalised over subsequent action
                    EF = np.sum(G_next_act,axis=1)
                    
                    K[index] = np.inner(softmax(EF),EF)
                indexes_L = list(idx_state_to_explore)

                # Expected free energy marginalised over states
                G[action,5] = G[action,5] + np.dot(K[indexes_L],Q[action][indexes_L])
        post_u = softmax(np.sum(G,axis=1))
        R = 0
        # Posterior over next state marginalised over subsequent action
        for action in range(U.shape[0]):
            R = R + post_u[action]*Q[action]
    return G,P