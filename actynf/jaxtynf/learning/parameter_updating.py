import random as ra
import numpy as np

from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.tree_util import tree_map
from jax import lax,vmap, jit

from .generalize_action_mapping import extrapolate_deltab

from ..shape_tools import to_source_space

# _______________________________________________________________________________________
# ________________________________ Dirichlet Weights Updating ___________________________
# _______________________________________________________________________________________

# Parameter updating terms given a hsitory of state inferences and observed values

# emissions :
def get_delta_a(hist_obs,hist_obs_bool,
                hist_qs,
                hidden_state_shape):
    def _delta_a_mod(o_mod,o_filter_mod):
        return jnp.reshape(jnp.einsum("ti,t,tj->ijt",o_mod,o_filter_mod,hist_qs).sum(axis=-1),(-1,)+hidden_state_shape) 
    return tree_map(_delta_a_mod,hist_obs,hist_obs_bool)

# transitions :

def get_delta_b_factor(hist_u,hist_qs,hist_u_bool):
    post_qs = hist_qs[1:,:]
    pre_qs = hist_qs[:-1,:]
    db = jnp.einsum("ti,tj,tu,t->ijut",post_qs,pre_qs,hist_u,hist_u_bool) 
                # For flattened transition mappings only !
    return db.sum(axis=-1)



def get_delta_b(hist_u_tree,hist_qs_tree,hist_u_bool,
                state_generalize_function=None,action_generalize_table=None,cross_action_extrapolation_coeff=0.1):
    
    # Dirichlet weight updating
    get_delta_b_factor_fitlered = partial(get_delta_b_factor,hist_u_bool=hist_u_bool)
    raw_db = tree_map(get_delta_b_factor_fitlered,hist_u_tree,hist_qs_tree)
    
    
    # And possible generalization
    if (type(state_generalize_function)!=list):
        state_generalize_function = [state_generalize_function for f in hist_u_tree]
    if (type(action_generalize_table)!=list):
        action_generalize_table = [action_generalize_table for f in hist_u_tree]
    if (type(cross_action_extrapolation_coeff) != list):
        cross_action_extrapolation_coeff = [cross_action_extrapolation_coeff for f in hist_u_tree]
    
    CLIP_EXTRAPOLATED_ACTIONS = False
    function_to_map = partial(extrapolate_deltab,option_clip=CLIP_EXTRAPOLATED_ACTIONS)       
        
    return tree_map(function_to_map,raw_db,
                    state_generalize_function,
                    action_generalize_table,cross_action_extrapolation_coeff)

# initial states :
def get_delta_d(hist_qs_tree):
    def _delta_d_factor(hist_qs_factor):
        return hist_qs_factor[0,:]
    
    return tree_map(_delta_d_factor,hist_qs_tree)


# Compute parameter update terms depending on options.
# This is operated at the trial level !
# meant to be vectorized along the trial dimension for the first 5 arguments.
def get_parameter_update(hist_obs_vect,hist_factor_action_vect,hist_u_vect,
                         hist_obs_bool,hist_factor_action_bool,
                        smoothed_posteriors,
                        Ns,Nu,
                        state_generalize_function=None,action_generalize_table=None,cross_action_extrapolation_coeff=0.1):
    r"""
    - Ns is the hidden state space shape.
    """
    
    # Warning ! When we missed observations or actions, we can't use it to 
    # update our parameters ! 
    
    # learning a is done in vectorized mode
    delta_a = get_delta_a(hist_obs_vect,hist_obs_bool,smoothed_posteriors,Ns)
    
    
    # For the state posteriors, factorize them after smoothing !
    factorized_smoothed_posteriors = vmap(lambda x : to_source_space(x,Ns))(smoothed_posteriors)
    
    delta_d = get_delta_d(factorized_smoothed_posteriors)
        
    delta_b = get_delta_b(hist_factor_action_vect,factorized_smoothed_posteriors,
                          hist_factor_action_bool,
                          state_generalize_function,action_generalize_table,cross_action_extrapolation_coeff)
    
    
    # c and e are not implemented yet 
    # (but should probably be guided by hierarchical processes anyways)... 
    delta_c = tree_map(lambda x : jnp.zeros((x.shape[-1])),hist_obs_vect)
    
    delta_e = hist_u_vect # jnp.zeros((Nu,))
        
    return delta_a,delta_b,delta_c,delta_d,delta_e


