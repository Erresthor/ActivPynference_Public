# -*- coding: utf-8 -*-

import numpy as np
from types import SimpleNamespace

from ...base.function_toolbox import normalize,spm_dot, nat_log,softmax
from ...base.miscellaneous_toolbox import isNone,flatten_last_n_dimensions,flexible_toString,flexible_print,flexible_copy

def spm_backwards(O,Q,A,B,K,T):
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

    L = Q[0]
    q = np.zeros(L.shape)
    for i in range(q.shape[0]):
        qi = np.copy(q)
        qi[i]=1
        for t in range(1,T):
            qi = np.dot(B[K[t-1]],qi)
            for modality in range(len(A)):
                flattened_A = flatten_last_n_dimensions(A[modality].ndim-1,A[modality])


                obs_vect = np.reshape(O[modality][:,t].T,(1,O[modality][:,t].shape[0]))
                state_given_observation = np.dot(obs_vect,flattened_A)
                        # State likelihood given observations at this tmstp
                individual_likelihood = np.dot(state_given_observation,qi)
                        # Posterior likelihood that i is the initial state
                        # given observations & previous state at time t

                L[i] = L[i]*individual_likelihood
                        # Complete Posterior over initial states given data at tmstps <=t
    return normalize(L)