import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp

from jax.tree_util import tree_map
from jax import lax,vmap,jit

from functools import partial
from itertools import product

from jax_toolbox import _normalize,_jaxlog
from jax_toolbox import _condition_on,_compute_A_conditional_logliks

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

@jax.jit
def compute_state_posterior(state_prior,new_obs,A):
    log_likelihoods = _compute_log_likelihood_multi_mod(A,new_obs)
    posterior,log_norm = _condition_on(state_prior,log_likelihoods)
    return posterior


