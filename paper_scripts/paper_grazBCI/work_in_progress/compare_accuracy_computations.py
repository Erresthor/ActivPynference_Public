

import numpy as np
import time
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import tree_map
from jax.scipy.special import gammaln,digamma,betaln
from tensorflow_probability.substrates import jax as tfp
from functools import partial

import optax

import matplotlib.pyplot as plt
import functools 

from actynf.jaxtynf.jax_toolbox import _jaxlog,_normalize












# From pymdp : 


def get_likelihood_single_modality(o_m, A_m, distr_obs=True):
    """Return observation likelihood for a single observation modality m"""
    if distr_obs:
        expanded_obs = jnp.expand_dims(o_m, tuple(range(1, A_m.ndim)))
        likelihood = (expanded_obs * A_m).sum(axis=0)
    else:
        likelihood = A_m[o_m]

    return likelihood

def compute_log_likelihood_single_modality(o_m, A_m, distr_obs=True):
    """Compute observation log-likelihood for a single modality"""
    return _jaxlog(get_likelihood_single_modality(o_m, A_m, distr_obs=distr_obs))

def compute_log_likelihood(obs, A, distr_obs=True):
    """ Compute likelihood over hidden states across observations from different modalities """
    result = tree_map(lambda o, a: compute_log_likelihood_single_modality(o, a, distr_obs=distr_obs), obs, A)
    ll = jnp.sum(jnp.stack(result), 0)

    return ll

def compute_log_likelihood_per_modality(obs, A, distr_obs=True):
    """ Compute likelihood over hidden states across observations from different modalities, and return them per modality """
    ll_all = tree_map(lambda o, a: compute_log_likelihood_single_modality(o, a, distr_obs=distr_obs), obs, A)

    return ll_all

def compute_accuracy(qs, obs, A):
    """ Compute the accuracy portion of the variational free energy (expected log likelihood under the variational posterior) """

    log_likelihood = compute_log_likelihood(obs, A)

    x = qs[0]
    for q in qs[1:]:
        x = jnp.expand_dims(x, -1) * q

    joint = log_likelihood * x
    return joint.sum()



# __________________________________________________________________________________
def _logexpect_dirichlet(dir_1,epsilon=1e-5):
    dir_1 = jnp.clip(dir_1, a_min=epsilon)
    return digamma(dir_1) - digamma(dir_1.sum(axis=0))

def _logexpect_dirichlet_list(list_a,epsilon=1e-5):
    def _logexpect_dirichlet_one_mod(list_a_mod):
        return _logexpect_dirichlet(list_a_mod,epsilon)
    logexpect_all_mods = tree_map(_logexpect_dirichlet_one_mod,list_a)
    return logexpect_all_mods
    # return jnp.stack(logexpect_all_mods)


def emission_term_multiple_factors(o,o_filter,qs,logA):
    """Compute the emission term of the VFE likelihood for
    the FTHMM. 

    Args:
        o (_type_): a list of Nmod tensor arrays of shape [Nout[mod]] encoding the emissions.
        qs (_type_):  a list of Nf tensor arrays of shape [Ns[f]] encoding the hiddens states.
        logA (_type_): a list of Nmod tensor arrays of shape [Nout[mod] x Ns(1) x Ns(2) x ... x Ns(Nf)] encoding emissions depedning on the hidden states.
    """
    # Reshape mods :
    _N_outcomes = len(logA)
    _latent_state_tuple = (1,)*(logA[0].ndim-1)
    
    
   
    def emission_term_one_mod(_logA_mod,_o_mod):    
        _Noutcomes_mod = _o_mod.shape[0]
        
        # reshape o to fit the matrix : 
        _reshaped_o_mod = jnp.reshape(_o_mod,(_Noutcomes_mod,)+_latent_state_tuple)
            
        joint_emission_logA = jnp.sum(_reshaped_o_mod*_logA_mod,axis=0)
        return joint_emission_logA
    
    # From pymdp (compute_accuracy method):
    x = qs[0]
    for q in qs[1:]:
        x = jnp.expand_dims(x, -1) * q
    
    # return tree_map(emission_term_one_mod,logA,o)
    log_all_modalities = jnp.stack(tree_map(emission_term_one_mod,logA,o))
    
    # We filter out the unseen observations here !
    o_filter_reshaped = jnp.reshape(o_filter,(_N_outcomes,)+_latent_state_tuple)
    filtered_log_modalities = (o_filter_reshaped*log_all_modalities).sum(axis=0)
    return filtered_log_modalities*x
    
    
def posterior_entropy(qs):
    def posterior_entropy_factor(_qs_f):
        return (_qs_f*_jaxlog(_qs_f)).sum(axis=-1)
    return jnp.stack(tree_map(posterior_entropy_factor,qs))

    
if __name__ == '__main__':
    # obs = [0, 1, 2, 2, 1]
    
    # qs = [jnp.array([0.3,0.5,0.2]),jnp.array([1.0,0.0,0.0,0.0])]
    # Ns = tuple([x.shape[0] for x in qs])
    
    # obs_vec = [ jax.nn.one_hot(o, nout) for o,nout in zip(obs,Noutcomes)]
    
    
    # print(A)
    
    # res = compute_log_likelihood(obs_vec, _normalize(A,tree=True))
    
    # print(res)
    # print([am.shape for am in A])
    # rez = emission_term_multiple_factors(obs_vec,qs,_logexpect_dirichlet_list(A))
    # print(rez)
    
    
    qu = jnp.array([0.5,0.2,0.3,0.0])
    
    U = jnp.array([
        [1,0],
        [0,0],
        [3,1],
        [2,1]
    ])
    
    Nactions,Nfactors = U.shape
    
    
    B = [jnp.ones((4,4,4)),jnp.ones((4,4,2))] # Factor transitions
    
    N_transitions = [b_f.shape[-1] for b_f in B]
    
    vecU = [jax.nn.one_hot(U[:,factor],N_transitions[factor]) for factor in range(Nfactors)]
    
    print(vecU)
    
    
    
    
    
    
    exit()
    
    Ntrials = 10
    Ntimesteps = 20
    
    keys  =[0,1,2,3,4]
    Noutcomes = [10,5,6,4,8]
    Nmodalities = len(Noutcomes)
    
    
    
    Ns = (5,4)
    
    import jax.random as jr
    obs_indexes = [jr.randint(jr.PRNGKey(key),(Ntrials,Ntimesteps),0,nout) for (key,nout) in zip(keys,Noutcomes)]
    print([o for o in obs_indexes])
    
    def _oh_indx(_o_mod,_noutm):
        return jax.nn.one_hot(_o_mod,_noutm)
    
    R = tree_map(_oh_indx,obs_indexes,Noutcomes)
    print(R)
    print([r.shape for r in R])

    o_filter = jnp.ones((Ntrials,Ntimesteps,Nmodalities))
    
    
    qs = [_normalize(jnp.ones((Ntrials,Ntimesteps,Ns_f)),axis=-1)[0] for Ns_f in Ns]
    print(qs)
    print([q.shape for q in qs])
    
    A = [jnp.ones((nout,) + Ns) / 3.0 for nout in Noutcomes]
    
    
    
    map_multiple_factors = vmap(vmap(emission_term_multiple_factors,in_axes=(0,0,0,None)),in_axes=(0,0,0,None))
    
    
    # v = vmap(vmap(posterior_entropy))(qs)
    # print(v.shape)
    
    print(posterior_entropy(qs).shape)
    
    
    # # obs = [0, 1, 2, 2, 1]
    # # Noutcomes = [10,5,6,4,8]
    
    # proposed_qs = [jnp.ones(Ns)]
    
    
    # def to_one_hot(_vec):
    #     return 
    
    # obs_vec = [ jax.nn.one_hot(o, nout) for o,nout in zip(obs,Noutcomes)]
    # # vec_outcomes_3_dims = []
    
    # # qs = [jnp.array([0.3,0.5,0.2]),jnp.array([1.0,0.0,0.0,0.0])]
    # # Ns = tuple([x.shape[0] for x in qs])
    
    
    
    # # A = [jnp.ones((nout,) + Ns) / 3.0 for nout in Noutcomes]
    # # print(A)
    
    # # res = compute_log_likelihood(obs_vec, _normalize(A,tree=True))
    
    # # print(res)
    # # print([am.shape for am in A])
    # # rez = emission_term_multiple_factors(obs_vec,qs,_logexpect_dirichlet_list(A))
    # # print(rez)