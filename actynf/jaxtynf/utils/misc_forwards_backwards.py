 # -*- coding: utf-8 -*-
"""
A Jax implementation of the spm_forwards function
________________________________________________________________

Created on 12/02/24

@author: Côme ANNICCHIARICO(come.annicchiarico@inserm.fr)

re-implementing dynamax hmm/inference.py functions
"""

import numpy as np
import jax.numpy as jnp

from jax_toolbox import _normalize,_jaxlog,_condition_on,_compute_A_conditional_logliks
from jax.nn import softmax
from jax import lax,vmap

def get_trans_mat(transition_matrix, transition_fn, t):
    if transition_fn is not None:
        return transition_fn(t)
    else:
        if transition_matrix.ndim == 3: # (T,K,K)
            return transition_matrix[t]
        else:
            return transition_matrix

def collapse_along_state(emission_function,state_distribution) : 
    """     
    emission_function(s) --> p(o|s)
    """
    return emission_function @ state_distribution



def forwards(D,B,A,
             observations,
             transition_fn=None):
    r"""Forwards filtering

    Transition matrix may be either 2D (if transition probabilities are fixed) or 3D
    if the transition probabilities vary over time. Alternatively, the transition
    matrix may be specified via `transition_fn`, which takes in a time index $t$ and
    returns a transition matrix.

    Args:
        D: $p(z_1 \mid u_1, \theta)$
        B: $p(z_{t+1} \mid z_t, u_t, \theta)$
        A: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.
        transition_fn: function that takes in an integer time index and returns a $K \times K$ transition matrix.

    Returns:
        filtered posterior distribution

    """
    log_likelihoods = _compute_A_conditional_logliks(A,observations)
    num_timesteps, num_states = log_likelihoods.shape

    def _step(carry, t):
        log_normalizer, predicted_probs = carry

        transition_matrix = get_trans_mat(B, transition_fn, t)
        ll = log_likelihoods[t]

        filtered_probs, log_norm = _condition_on(predicted_probs, ll)
        
        predicted_probs_next = transition_matrix @ filtered_probs

        log_normalizer += log_norm
        return (log_normalizer, predicted_probs_next), (filtered_probs, predicted_probs)

    carry = (0.0, D)
    (log_normalizer, _), (filtered_probs, predicted_probs) = lax.scan(_step, carry, jnp.arange(num_timesteps))

    # post = HMMPosteriorFiltered(marginal_loglik=log_normalizer,
    #                             filtered_probs=filtered_probs,
    #                             predicted_probs=predicted_probs)
    return log_normalizer,filtered_probs,predicted_probs



def backwards(B,A,
    observations,
    transition_fn=None):
    r"""Run the filter backwards in time. This is the second step of the forward-backward algorithm.

    Transition matrix may be either 2D (if transition probabilities are fixed) or 3D
    if the transition probabilities vary over time. Alternatively, the transition
    matrix may be specified via `transition_fn`, which takes in a time index $t$ and
    returns a transition matrix.

    Args:
        B: $p(z_{t+1} \mid z_t, u_t, \theta)$
        A: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.
        transition_fn: function that takes in an integer time index and returns a $K \times K$ transition matrix.

    Returns:
        marginal log likelihood and backward messages.

    """
    log_likelihoods = _compute_A_conditional_logliks(A,observations)
    num_timesteps, num_states = log_likelihoods.shape

    def _step(carry, t):
        log_normalizer, backward_pred_probs = carry

        transition_matrix = get_trans_mat(B, transition_fn, t)
        ll = log_likelihoods[t]

        # Condition on emission at time t, being careful not to overflow.
        backward_filt_probs, log_norm = _condition_on(backward_pred_probs, ll)
        # Update the log normalizer.
        log_normalizer += log_norm
        # Predict the next state (going backward in time).
        next_backward_pred_probs = transition_matrix.T @ backward_filt_probs
        return (log_normalizer, next_backward_pred_probs), backward_pred_probs

    carry = (0.0, jnp.ones(num_states))
    (log_normalizer, _), rev_backward_pred_probs = lax.scan(_step, carry, jnp.arange(num_timesteps)[::-1])
    backward_pred_probs = rev_backward_pred_probs[::-1]
    return log_normalizer, backward_pred_probs

def raw_smooth(D,B,A,observations):
    b_mll,fwd_probs,_ = forwards(D,B,A,observations)
    f_mll,bkw_probs = backwards(B,A,observations)

    # Compute smoothed probabilities
    smoothed_probs = fwd_probs * bkw_probs
    norm = smoothed_probs.sum(axis=1, keepdims=True)
    smoothed_probs /= norm

    return smoothed_probs

def better_smooth(D,B,A,observations,
                transition_fn=None):

    # The forward pass goes on as planned
    f_mll,fwd_probs,fwd_preds = forwards(D,B,A,observations)

    # Recompute the conditionnal log-kilelihoods of states given observations
    #    = log p(o_t|s_t) for t in [0,T]
    log_likelihoods = _compute_A_conditional_logliks(A,observations)
    num_timesteps, num_states = log_likelihoods.shape
    

    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        smoothed_probs_next = carry
        t, fwd_prob, fwd_pred = args

        transition_matrix = get_trans_mat(B, transition_fn, t)

        # Fold in the next state (Eq. 8.2 of Saarka, 2013)
        # If hard 0. in predicted_probs_next, set relative_probs_next as 0. to avoid NaN values
        relative_probs_next = jnp.where(jnp.isclose(fwd_pred, 0.0), 0.0,
                                        smoothed_probs_next / fwd_pred)
        
        smoothed_probs = fwd_prob * (B.T @ relative_probs_next)
        smoothed_probs /= smoothed_probs.sum()

        return smoothed_probs, smoothed_probs
    

    carry = fwd_probs[-1]
    args = (jnp.arange(num_timesteps - 2, -1, -1), fwd_probs[:-1][::-1], fwd_preds[1:][::-1])

    _, rev_smoothed_probs = lax.scan(_step, carry, args)

    # Reversed :
    smoothed_probs = jnp.vstack([rev_smoothed_probs[::-1], fwd_preds[-1]])

    return smoothed_probs

def viterbi(D,B,A,observations,
            transition_fn=None):
    log_likelihoods = _compute_A_conditional_logliks(A,observations)
    num_timesteps, num_states = log_likelihoods.shape
    
    # Run the backward pass
    def _backward_pass(best_next_score, t):
        transition_matrix = get_trans_mat(B, transition_fn, t)

        scores = jnp.log(transition_matrix.T) + best_next_score + log_likelihoods[t + 1]
        best_next_state = jnp.argmax(scores, axis=1)
        best_next_score = jnp.max(scores, axis=1)
        return best_next_score, best_next_state
    num_states = log_likelihoods.shape[1]
    best_second_score, rev_best_next_states = lax.scan(
        _backward_pass, jnp.zeros(num_states), jnp.arange(num_timesteps - 2, -1, -1)
    )
    best_next_states = rev_best_next_states[::-1]

    # Run the forward pass
    def _forward_pass(state, best_next_state):
        next_state = best_next_state[state]
        return next_state, next_state

    first_state = jnp.argmax(jnp.log(D) + log_likelihoods[0] + best_second_score)
    _, states = lax.scan(_forward_pass, first_state, best_next_states)

    # for t in range(num_timesteps,-1,-1):
    #     print(jnp.log(B) + 0 + log_likelihoods[t])
    #     print(jnp.argmax(scores, axis=0))

    return jnp.concatenate([jnp.array([first_state]), states])

def compute_transitions_probabilities(B,smoothed_posteriors,
            transition_fn=None):

    def transition_probabilities(t):
        trans_mat = get_trans_mat(B,transition_fn,t)  # i,j <=> to_state x from_state
        return jnp.einsum('i,j->ij',smoothed_posteriors[t+1],smoothed_posteriors[t])
        # return jnp.einsum('i,kl,j->ij',smoothed_posteriors[t],trans_mat,smoothed_posteriors[t+1])

    transition_probabilities = vmap(transition_probabilities)(jnp.arange(smoothed_posteriors.shape[0]-1))
    return transition_probabilities


# TODO : implement the same testing functions as in :
# https://github.com/probml/dynamax/blob/main/dynamax/hidden_markov_model/inference_test.py#L176 

