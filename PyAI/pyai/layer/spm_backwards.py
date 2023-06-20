# -*- coding: utf-8 -*-

import numpy as np
from types import SimpleNamespace

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
                L[i] = L[i] + nat_log(individual_likelihood)
                        # Complete Posterior over initial states given data at tmstps <=t
    return softmax(L)


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
    #     q = np.zeros(L.shape)
    print("€€€€€€€€€€€€€€")
    print(L)
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
        print(L)
        print(np.round(L,2))
    return normalize(L)