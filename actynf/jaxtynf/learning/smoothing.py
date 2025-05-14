from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.tree_util import tree_map
from jax import lax,vmap, jit

from ..layer_infer_state import get_log_likelihood_all_timesteps
from ..jax_toolbox import _condition_on,_normalize

# E-step : given parameters, what is the most likely sequence of states ?
def forwards_pass(hist_obs_vect,hist_u_vect,
                  hist_obs_bool,
                  vec_a,vec_b,vec_d):
    r"""Forwards filtering for a history of emissions and actions defined across a single trial.
    Args:
        Data :
        - hist_obs_vect is a list of tensors (one tensor per modality). Each tensor is of shape (Ntimesteps x Noutcomes(modality))
        - hist_obs_bool is a list of tensors (one tensor per modality). Each tensor is of shape (Ntimesteps) and has value
            1.0 if the value was observed and 0.0 if it was not.
        - hist_u_vect is a single tensor of shape ((Ntimesteps-1) x Nactions)
        Parameters :
        - vec_d: flattened initial state $p(z_1 \mid \theta)$ ; tensor of shape (Nstates)
        - vec_b: flattened transition matrix $p(z_{t+1} \mid z_t, u_t, \theta)$ ; tensor of shape (Nstates x Nstates x Nactions)
        - vec_a: flattened list of emission mappings across modalities $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$  ; list of tensors of shape (Noutcomes(modality) x Nstates)
        
    Returns:
        marginal log likelihood and filtered posterior distribution ; tensor of shape (Ntimesteps x Nstates)

    """        
    # Compute the log likelihooods of each emission (if they were observed)
    _,logliks = get_log_likelihood_all_timesteps(hist_obs_vect,vec_a,hist_obs_bool)
    num_timesteps, num_states = logliks.shape
    
    def _step(carry, t):
        carry_log_norm,prior_this_timestep = carry
        
        forward_filt_probs, log_norm = _condition_on(prior_this_timestep, logliks[t])
        
        carry_log_norm = carry_log_norm + log_norm
        
        # Predict the next state (going forward in time).
        transition_matrix = jnp.einsum("iju,u->ij",vec_b,hist_u_vect[t-1,...])
                # If hist_u_vect was not observed, it may be replaced by a flat mapping ? subjects habits ? Infered ?
                # See VFE minimization with flexible transitions :) : actynf.jax_methods.utils.misc_vfe_minimization_action_mdp.py

        prior_next_timestep = transition_matrix @ forward_filt_probs
                
        return (carry_log_norm,prior_next_timestep),(forward_filt_probs,log_norm)

    init_carry = (jnp.array([0.0]),vec_d)
    (sum_log_norm,_),(forward_pred_probs,elbo_history) = lax.scan(_step, init_carry, jnp.arange(num_timesteps))
    
    return sum_log_norm,elbo_history,forward_pred_probs

def backwards_pass(hist_obs_vect,hist_u_vect,
                  hist_obs_bool,
                  vec_a,vec_b):
    r"""Run the filter backwards in time. This is the second step of the forward-backward algorithm.

    Args:
        Data :
        - hist_obs_vect is a list of tensors (one tensor per modality). Each tensor is of shape (Ntimesteps x Noutcomes(modality))
        - hist_obs_bool is a list of tensors (one tensor per modality). Each tensor is of shape (Ntimesteps) and has value
            1.0 if the value was observed and 0.0 if it was not.
        - hist_u_vect is a single tensor of shape ((Ntimesteps-1) x Nactions)
        Parameters :
        - vec_b: flattened transition matrix $p(z_{t+1} \mid z_t, u_t, \theta)$ ; tensor of shape (Nstates x Nstates x Nactions)
        - vec_a: flattened list of emission mappings across modalities $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$  ; list of tensors of shape (Noutcomes(modality) x Nstates)
        
    Returns:
        filtered posterior distribution ; tensor of shape (Ntimesteps x Nstates)
    Returns:
        marginal log likelihood and backward messages ; tensor of shape (Ntimesteps x Nstates)

    """     
    # Compute the log likelihooods of each emission (if they were observed)
    _,logliks = get_log_likelihood_all_timesteps(hist_obs_vect,vec_a,hist_obs_bool)
    
    num_timesteps, num_states = logliks.shape

    def _step(carry, t):
        carry_log_norm,prior_this_timestep = carry
        
        backward_filt_probs, log_norm = _condition_on(prior_this_timestep, logliks[t])
        
        carry_log_norm = carry_log_norm + log_norm
        
        # Predict the next (previous) state (going backward in time).
        transition_matrix = jnp.einsum("iju,u->ij",vec_b,hist_u_vect[t-1,...])
        prior_previous_timestep = transition_matrix.T @ backward_filt_probs
                
        return (carry_log_norm,prior_previous_timestep),(backward_filt_probs,log_norm)

    init_carry = (jnp.array([0.0]),jnp.ones(num_states))
    (sum_log_norm,_),(backward_pred_probs_rev,elbo_history) = lax.scan(_step, init_carry, jnp.arange(num_timesteps)[::-1])
    
    backward_pred_probs = backward_pred_probs_rev[::-1]
    return sum_log_norm,elbo_history, backward_pred_probs

def smooth_trial(hist_obs_vect,hist_u_vect,
                  hist_obs_bool,
                  vec_a,vec_b,vec_d,
                  filter_type="two_filter",hist_qs=None):
    r"""Forwards-backwards filtering for a history of emissions and actions defined across a single trial.
    Args:
        Data :
        - hist_obs_vect is a list of tensors (one tensor per modality). Each tensor is of shape (Ntimesteps x Noutcomes(modality))
        - hist_obs_bool is a list of tensors (one tensor per modality). Each tensor is of shape (Ntimesteps) and has value
            1.0 if the value was observed and 0.0 if it was not.
        - hist_u_vect is a single tensor of shape ((Ntimesteps-1) x Nactions)
        Parameters :
        - vec_d: flattened initial state $p(z_1 \mid \theta)$ ; tensor of shape (Nstates)
        - vec_b: flattened transition matrix $p(z_{t+1} \mid z_t, u_t, \theta)$ ; tensor of shape (Nstates x Nstates x Nactions)
        - vec_a: flattened list of emission mappings across modalities $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$  ; list of tensors of shape (Noutcomes(modality) x Nstates)
        
    Returns:
        smoothed posterior distribution given the proposed parameters ; tensor of shape (Ntimesteps x Nstates)
    """    
    if filter_type=="two_filter":
        forward_sum_elbo,forward_hist_elbo,forwards_smooths = forwards_pass(
                    hist_obs_vect,hist_u_vect,hist_obs_bool,
                    vec_a,vec_b,vec_d
        )
        backward_sum_elbo,backward_hist_elbo,backwards_smooths = backwards_pass(
                hist_obs_vect,hist_u_vect,hist_obs_bool,
                vec_a,vec_b
        )
        smoothed_posterior,_ = _normalize(forwards_smooths*backwards_smooths,axis=-1)

        sum_elbo = forward_sum_elbo
        hist_elbo = forward_hist_elbo
    elif filter_type=="one_filter":
        forwards_smooths = hist_qs
        
        # Assume that the forward filtering was done already during trial
        backward_sum_elbo,backward_hist_elbo,backwards_smooths = backwards_pass(
                hist_obs_vect,hist_u_vect,hist_obs_bool,
                vec_a,vec_b
        )
        smoothed_posterior,_ = _normalize(forwards_smooths*backwards_smooths,axis=-1)
        
        sum_elbo = backward_sum_elbo
        hist_elbo = backward_hist_elbo
    else :
        raise NotImplementedError("The developer is lazy , this has not been done yet :'(")

    return smoothed_posterior,(sum_elbo,hist_elbo)

def smooth_trial_window(hist_obs_vect,hist_u_vect,
                  hist_obs_bool,
                  vec_a,vec_b,vec_d):
    r"""Forwards-backwards filtering for a history of emissions and actions defined across a window of several trials (leading dimension).
    Args:
        Data :
        - hist_obs_vect is a list of tensors (one tensor per modality). Each tensor is of shape (Ntrials x Ntimesteps x Noutcomes(modality))
        - hist_obs_bool is a list of tensors (one tensor per modality). Each tensor is of shape (Ntrials x Ntimesteps) and has value
            1.0 if the value was observed and 0.0 if it was not.
        - hist_u_vect is a single tensor of shape (Ntrials x (Ntimesteps-1) x Nactions)
        Parameters :
        - vec_d: flattened initial state $p(z_1 \mid \theta)$ ; tensor of shape (Nstates)
        - vec_b: flattened transition matrix $p(z_{t+1} \mid z_t, u_t, \theta)$ ; tensor of shape (Nstates x Nstates x Nactions)
        - vec_a: flattened list of emission mappings across modalities $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$  ; list of tensors of shape (Noutcomes(modality) x Nstates)
        
    Returns:
        smoothed posterior distribution given the proposed parameters ; tensor of shape (Ntrials x Ntimesteps x Nstates)
    """  
    return vmap(smooth_trial,in_axes=(0,0,0,None,None,None))(hist_obs_vect,hist_u_vect,hist_obs_bool,vec_a,vec_b,vec_d)











