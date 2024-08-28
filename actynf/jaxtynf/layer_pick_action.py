import jax.numpy as jnp
import jax.random as jr
import jax

from numpyro import plate,sample,deterministic
import numpyro.distributions as distr

from jax.tree_util import tree_map
from jax import vmap
from jax import jit

from functools import partial

from .jax_toolbox import _jaxlog


### Sample an action from the posterior --------------------------------------------------------------------------
@jax.jit
def alpha_weight(raw_posterior,alpha):
    alpha_weighted_posterior = jax.nn.softmax(_jaxlog(raw_posterior) * alpha)
    return alpha_weighted_posterior

# More compacted methods using an action_selection_options dictionnary
def sample_action(qpi,action_select_option,rng_key=None):
    alpha = action_select_option["alpha"]
    selection_method=action_select_option["method"]
    
    Np = qpi.shape[-1]
    if selection_method == "deterministic":
        action_idx = jnp.argmax(qpi)
        action_dist = jax.nn.one_hot(action_idx,Np)
        action_vect = jax.nn.one_hot(action_idx,Np)
    elif (selection_method == "stochastic") or (selection_method == "stochastic_alpha"):
        action_dist = alpha_weight(qpi,alpha)
        action_idx = jr.categorical(rng_key, _jaxlog(action_dist))
        action_vect = jax.nn.one_hot(action_idx,Np)
    else : 
        raise NotImplementedError("Action selection method not implemented : '" + str(selection_method) + "'.")
    return action_dist,action_idx,action_vect

def sample_action_pyro(qpi,action_select_option,observed_action=None):
    """ 
    When lost about shapes in numpyro :
    https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/
    """
    alpha = action_select_option["alpha"]
    selection_method=action_select_option["method"]
    
    Np = qpi.shape[-1]
    if observed_action != None:
        assert qpi.shape[:-1]==observed_action.shape,"Shape mismatch in sample_action_pyro"
    
    if selection_method == "deterministic":
        action_idx = jnp.argmax(qpi,axis=-1)
        action_dist = jax.nn.one_hot(action_idx,Np,axis=-1)
        action_vect = jax.nn.one_hot(action_idx,Np,axis=-1)
        deterministic("action_t",action_idx)
    elif (selection_method == "stochastic") or (selection_method == "stochastic_alpha"):
        action_dist = alpha_weight(qpi,alpha) # Along -1th axis
        # One sample of a multivariate ((Ntrials x ) Ntimesteps-1 x ) Np distribution
        action_idx = sample("actions",distr.Categorical(probs=action_dist).to_event(action_dist.ndim-1),obs=observed_action)
        action_vect = jax.nn.one_hot(action_idx,Np)
    else : 
        raise NotImplementedError("Action selection method not implemented : '" + str(selection_method) + "'.")
    return action_dist,action_idx,action_vect
