import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp

from jax.tree_util import tree_map
from jax import lax,vmap,jit

from functools import partial
from itertools import product

from actynf.jaxtynf.jax_toolbox import _normalize,_jaxlog
from actynf.jaxtynf.jax_toolbox import _condition_on,_compute_A_conditional_logliks

import tensorflow_probability.substrates.jax.distributions as tfd

# Log likelihoods assuming matrices :
def _compute_log_likelihood_multi_mod(A, o_d, timedim=False):
    """ 
    Compute the log likelihood of the list of distribution o_d.
    Assuming A_m is a matrix of shape Noutcome[modality] x num_states and o_d list of dims of shape Noutcome[modality]
    IF timedim is set to True, o_d is assumed to be defined along the whole Trial dimension, and the function will map 
    likelihoods across this dimension. --> poorly defined ? :'(
    """
    def _compute_log_likelihood_single_mod(A_m, o_d_m):
        """ 
        Compute the log likelihood of the distribution o_d_m.
        Assuming A_m is a matrix of shape Noutcome[modality] x num_states  and o_d_m of shape Noutcome[modality]
        """
        func = lambda o : _jaxlog(jnp.einsum("ij,i->j",A_m,o))
        if timedim:
            #o_d_m is of shape Noutcome[modality] x T, we map over T !
            T =  o_d_m.shape[-1]
            timefunc = lambda t: func(o_d_m[:,t])
            return vmap(timefunc)(jnp.arange(0,T,1))
        else :
            return func(o_d_m)
        
    r = tree_map(_compute_log_likelihood_single_mod,A,o_d)
    return jnp.stack(r).sum(axis=0)

# New log likelihood computation with filters to indicate unseen observations :
# Utils :
def get_log_likelihood_one_observation(o_m,a_m,obs_m_filter=None): 
    # For 2D observation matrices (flattened state space)
    if obs_m_filter is None:
        return _jaxlog(jnp.einsum("ij,i->j",a_m,o_m))
    return _jaxlog(jnp.einsum("ij,i->j",a_m,o_m))*obs_m_filter

def get_log_likelihood_one_timestep(o,a,obs_filter=None):
    """_summary_

    Args:
        o (_type_): A list of Nmodalities tensors of shape Noutcomes[mod]
        a (_type_): A list of Nmodalities likelihood mappings of shape Noutcomes[mod] x Ns
        obs_filter (_type_, optional): A list of binary tensors indivating if this observation was indeed seen by the agent. Defaults to None (all observations were seen).

    Returns:
        ll_each_mod: A  Nmodalities x  Ns tensor of log likelihoods for this timestep, given a.
    """
    if obs_filter is None :
        ll_one_mod = partial(get_log_likelihood_one_observation,obs_m_filter=None)
        ll_each_mod = jnp.stack(tree_map(ll_one_mod,o,a),axis=-2)
    else :
        ll_each_mod = jnp.stack(tree_map(get_log_likelihood_one_observation,o,a,obs_filter),axis=-2)
    
    return ll_each_mod, jnp.sum(ll_each_mod,axis=-2)

def get_log_likelihood_all_timesteps(o,a,obs_filter=None):
    """ 
    Compute the log likelihood of the list of distribution o given a.
    Assuming a_m is a matrix of shape Noutcome[modality] x num_states and o_d list of dims of shape  T x Noutcome[modality]
    Each element of o is assumed to be defined along a number of timesteps, and the function will map 
    likelihoods across this dimension.
    Obs_filters is a list of binary (1 or 0) values indicating wether or not the observation for this modality was available to the
    agent.
    """
    ll_each_mod,ll_all_mod = vmap(get_log_likelihood_one_timestep,in_axes=(0,None,0))(o,a,obs_filter) # T x Nmodalities x  Ns,T x Ns
    return ll_each_mod,ll_all_mod



@jax.jit
def compute_state_posterior(state_prior,new_obs,A,obs_filter=None):
    # TODO : introduce more complex state inference methods : Fixed Point Iteration, Variational Filtering, Message Passing 
    # (If we want to use factorized representations during planning / hierarchical representations ?)
    
    # Simple bayesian filter :
    ll_each_mod,ll_all_mod = get_log_likelihood_one_timestep(new_obs,A,obs_filter)
    posterior,log_norm = _condition_on(state_prior,ll_all_mod)
    
    # Here, log_norm is the ELBO / negative FE of our model given the data
    return posterior,log_norm


if __name__ == "__main__":
    import numpy as np
    
    raw_a = [np.zeros((2,2,3)),np.zeros((3,2,3))]

    for s in range(3):
        raw_a[0][:,:,s] = np.array([
            [0.8,0.3],
            [0.2,0.7]
        ])
        raw_a[1][:,:,s] = ([
            [1.0,0.0],
            [0.0,1.0],
            [1.0,1.0]
        ])
    vec_a = [_normalize(jnp.reshape(a_m,(a_m.shape[0],-1)))[0] for a_m in raw_a]    
    
    filters = [jnp.array([1.0,1.0,1.0]),jnp.array([1.0,0.0,0.0])]
    
    o_d_1 = [jnp.array([
        [0.9,0.1],
        [0.8,0.2],
        [0.0,1.0]
    ]),jnp.array([
        [0.9,0.1,0.0],
        [1.0,1.0,1.0],
        [1.0,1.0,1.0]
    ])]
    obs = [_normalize(o_m,-1)[0] for o_m in o_d_1]
        
    ll_each_mod,ll_all_mod = get_log_likelihood_all_timesteps(obs,vec_a,filters)
    print(ll_all_mod)
    
    
    t = 2
    obs_one_timestep = [o[t,...] for o in obs]
    print(obs_one_timestep)
    
    prior_qs = _normalize(jnp.ones(6,))[0]
    
    prior_qs = jax.nn.one_hot(2,6)
    
    qs,F = compute_state_posterior(prior_qs,obs_one_timestep,vec_a,obs_filter=None)
    
    print(qs)
    print(F)