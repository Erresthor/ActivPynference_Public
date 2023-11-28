# -*- coding: utf-8 -*-

import numpy as np

from ..base.function_toolbox import normalize,spm_dot, nat_log,softmax
from ..base.miscellaneous_toolbox import flexible_copy

def spm_backwards(O,Q,flattenedA,B):
    # % Backwards smoothing to evaluate posterior over initial states
    # %--------------------------------------------------------------------------
    # % O{g}   - cell array of outcome probabilities for modality g
    # % Q{t}   - posterior expectations over vectorised hidden states
    # % A{g}   - likelihood mappings from hidden states
    # % B{k}   - belief propagators (action dependent probability transitions)
    # % u{t}   - posterior expectations over actions
    # % T      - time horizon
    # %
    # % L      - posterior over initial states
    # %
    # %  This subroutine evaluate the posterior over initial states using a
    # %  backwards algorithm; namely, by evaluating the likelihood of each
    # %  initial state, given subsequent outcomes, under posterior expectations
    # %  about state transitions.


    # % initialise to posterior and accumulate likelihoods for each initial state
    # %--------------------------------------------------------------------------

    # NOTE : It seems that for high values of T (trials with a high number of timesteps), the 
    # overall likelihood gets very low :/ 
    
    # Introducing a log-likelihood scheme :
    L = nat_log(flexible_copy(Q[:,0])) # Kronecker form of initial states likelihood
    #     q = np.zeros(L.shape)
    for i in range(L.shape[0]):  # For all possible states
        print(i)
        qi = np.zeros(L.shape)
        qi[i]=1 # If initial state was i
        for t in range(1,len(B)+1):
            qi = np.dot(B[t-1],qi) # Then at time t it would be :
            for modality in range(len(flattenedA)):
                obs_vect = np.reshape(O[modality][:,t].T,(1,O[modality][:,t].shape[0]))
                state_given_observation = np.dot(obs_vect,flattenedA[modality])
                        # State likelihood given observations at this tmstp
                individual_likelihood = np.dot(state_given_observation,qi)
                        # Posterior likelihood that i is the initial state
                        # given observations & previous state at time t
                L[i] = L[i] + nat_log(individual_likelihood)
                        # Complete Posterior over initial states given data at tmstps <=t
    return softmax(L)

def backward_state_posterior_estimation(O,Q,flattenedA,B):
    # % Backwards smoothing to evaluate posterior over hidden states
    # %--------------------------------------------------------------------------
    # % O{g}   - cell array of outcome probabilities for modality g
    # % Q{t}   - posterior expectations over vectorised hidden states
    # % A{g}   - likelihood mappings from hidden states
    # % B{k}   - belief propagators (action dependent probability transitions)
    # % u{t}   - posterior expectations over actions
    # % T      - time horizon
    # %
    # % P      - posterior over hidden states 
    # %
    # %  This subroutine evaluate the posterior over initial states using a
    # %  backwards algorithm; namely, by evaluating the likelihood of each
    # %  initial state, given subsequent outcomes, under posterior expectations
    # %  about state transitions.
    # %--------------------------------------------------------------------------
    Nmod = len(O)
    P_priors = flexible_copy(Q)
    P_posteriors = flexible_copy(Q)
    for t in range(len(B),-1,-1):
        prior_P = P_priors[:,t]
        
        # L is the posterior over hidden states at time t based on likelihood (A & O)
        L = 1
        for modality in range (Nmod):
            L = L * spm_dot(flattenedA[modality],O[modality][:,t]) 
        post_unnormalized = L*prior_P # posterior unnormalised
        # post_unnormalized is the posterior over hidden states at the current time t based on priors
        # and evidence
        F = nat_log(np.sum(post_unnormalized))

        smoothed_posterior =normalize(post_unnormalized) # P(s|o) = P(o|s)P(s)
        
        if (t>0):
            P_priors[:,t-1] = np.dot(smoothed_posterior,B[t-1])
        P_posteriors[:,t] = smoothed_posterior
    return P_posteriors

def old_spm_backwards(O,Q,flattenedA,B):
    # % Backwards smoothing to evaluate posterior over initial states
    # %--------------------------------------------------------------------------
    # % O{g}   - cell array of outcome probabilities for modality g
    # % Q{t}   - posterior expectations over vectorised hidden states
    # % A{g}   - likelihood mappings from hidden states
    # % B{k}   - belief propagators (action dependent probability transitions)
    # % u{t}   - posterior expectations over actions
    # % T      - time horizon
    # %
    # % L      - posterior over initial states
    # %
    # %  This subroutine evaluate the posterior over initial states using a
    # %  backwards algorithm; namely, by evaluating the likelihood of each
    # %  initial state, given subsequent outcomes, under posterior expectations
    # %  about state transitions.


    # % initialise to posterior and accumulate likelihoods for each initial state
    # %--------------------------------------------------------------------------
    L = flexible_copy(Q[:,0]) # Kronecker form of initial states likelihood
    for i in range(L.shape[0]):  # For all possible states
        qi = np.zeros(L.shape)
        qi[i]=1 # If initial state was i
        for t in range(1,len(B)+1):
            qi = np.dot(B[t-1],qi) # Then at time t it would be :
        #     print(qi)
            for modality in range(len(flattenedA)):
                obs_vect = np.reshape(O[modality][:,t].T,(1,O[modality][:,t].shape[0]))
                state_given_observation = np.dot(obs_vect,flattenedA[modality])
                        # State likelihood given observations at this tmstp
                individual_likelihood = np.dot(state_given_observation,qi)
                        # Posterior likelihood that i is the initial state
                        # given observations & previous state at time t
                L[i] = L[i]*individual_likelihood
                        # Complete Posterior over initial states given data at tmstps <=t
    return normalize(L)