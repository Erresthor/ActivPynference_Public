import os
import random as ra
import numpy as np
import time
import copy
import matplotlib.pyplot as plt

from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.tree_util import tree_map
from jax import lax,vmap,jit

import arviz as az
import corner 

from numpyro import plate,sample,deterministic
import numpyro.distributions as distr
import numpyro 
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive    
    
from .jax_toolbox import _normalize,convert_to_one_hot_list,_swapaxes
from .layer_pick_action import sample_action_pyro

from .layer_options import DEFAULT_PLANNING_OPTIONS,DEFAULT_LEARNING_OPTIONS,DEFAULT_ACTION_SELECTION_OPTIONS

from .layer_trial import synthetic_trial
from .layer_trial import empirical as empirical_trial

from .layer_learn import learn_after_trial

from actynf.jaxtynf.shape_tools import to_log_space,vectorize_weights,get_vectorized_novelty


def training_step(trial_rng_key,T,
            pa,pb,pd,c,e,u,
            A_vec,B_vec,D_vec,
            planning_options = DEFAULT_PLANNING_OPTIONS,
            action_selection_options = DEFAULT_ACTION_SELECTION_OPTIONS,
            learning_options = DEFAULT_LEARNING_OPTIONS):
    """_summary_

    Args:
        trial_rng_key (_type_): _description_
        T (_type_): _description_
        Th (_type_): _description_
        pa (_type_): _description_
        pb (_type_): _description_
        pd (_type_): _description_
        c (_type_): _description_
        e (_type_): _description_
        A_vec (_type_): Vectorized process emission rule
        B_vec (_type_): Vectorized process transition rule
        D_vec (_type_): Vectorized process initial state rule
        selection_method (str, optional): _description_. Defaults to "stochastic".
        alpha (int, optional): _description_. Defaults to 16.
        planning_options (_type_, optional): _description_. Defaults to DEFAULT_PLANNING_OPTIONS.
        learn_dictionnary (_type_, optional): _description_. Defaults to DEFAULT_LEARNING_OPTIONS.

    Returns:
        _type_: _description_
    """
    # Vectorize the model weights : 
    trial_a,trial_b,trial_d = vectorize_weights(pa,pb,pd,u)
    trial_c,trial_e = to_log_space(c,e)
    trial_a_nov,trial_b_nov = get_vectorized_novelty(pa,pb,u,compute_a_novelty=True,compute_b_novelty=True)
    
    
    # T timesteps happen below : 
    [obs_darr,obs_arr,obs_vect_arr,
        true_s_darr,true_s_arr,true_s_vect_arr,
        u_d_arr,u_arr,u_vect_arr,
        qs_arr,qpi_arr,efes] = synthetic_trial(trial_rng_key,T,
                    A_vec,B_vec,D_vec,
                    trial_a,trial_b,trial_c,trial_d,trial_e,
                    trial_a_nov,trial_b_nov,
                    planning_options=planning_options,
                    action_selection_options=action_selection_options)
    
    # Then, we update the parameters of our HMM model at this level
    # We use the raw weights here !
    a_post,b_post,c_post,d_post,e_post,qs_post = learn_after_trial(obs_vect_arr,qs_arr,u_vect_arr,
                                            pa,pb,c,pd,e,u,
                                            method = learning_options["method"],
                                            learn_what = learning_options["bool"],
                                            learn_rates=learning_options["learning_rates"],
                                            forget_rates=learning_options["forgetting_rates"],
                                            generalize_state_function=learning_options["state_generalize_function"],
                                            generalize_action_table=learning_options["action_generalize_table"],
                                            cross_action_extrapolation_coeff=learning_options["cross_action_extrapolation_coeff"],
                                            em_iter = learning_options["em_iterations"])
    
    return_tuple = ( obs_darr,obs_arr,obs_vect_arr,
                    true_s_darr,true_s_arr,true_s_vect_arr,
                    u_d_arr,u_arr,u_vect_arr,
                    qs_arr,qs_post,qpi_arr,efes,
                    a_post,b_post,c_post,d_post,e_post)
    
    return return_tuple

# Very fast methods
def synthetic_training(rngkey,
            Ntrials,T,
            A,B,D,U,
            a0,b0,c,d0,e,u,
            planning_options = DEFAULT_PLANNING_OPTIONS,
            action_selection_options = DEFAULT_ACTION_SELECTION_OPTIONS,
            learning_options = DEFAULT_LEARNING_OPTIONS):
    normA,normB,normD = vectorize_weights(A,B,D,U)
        # These weights are the same across the whole training

    def _scan_training(carry,key):
        key,trial_key = jr.split(key)
        
        pa,pb,pd = carry
        
        # T timesteps happen below : 
        ( obs_darr,obs_arr,obs_vect_arr,
        true_s_darr,true_s_arr,true_s_vect_arr,
        u_d_arr,u_arr,u_vect_arr,
        qs_arr,qs_post,qpi_arr,efes,
        a_post,b_post,c_post,d_post,e_post) = training_step(trial_key,T,
            pa,pb,pd,c,e,u,
            normA,normB,normD,
            planning_options = planning_options,
            action_selection_options=action_selection_options,
            learning_options = learning_options)
        
        # a_post,b_post,d_post = pa,pb,pd
        return (a_post,b_post,d_post),(
                    obs_darr,obs_arr,obs_vect_arr,
                    true_s_darr,true_s_arr,true_s_vect_arr,
                    u_d_arr,u_arr,u_vect_arr,
                    qs_arr,qs_post,qpi_arr,efes,
                    a_post,b_post,c_post,d_post,e_post)
        
    
    next_keys = jr.split(rngkey, Ntrials)
    (final_a,final_b,final_d), (
        all_obs_darr,all_obs_arr,all_obs_vect_arr,
        all_true_s_darr,all_true_s_arr,all_true_s_vect_arr,
        all_u_d_arr,all_u_arr,all_u_vect_arr,
        all_qs_arr,all_qs_post,all_qpi_arr,efes_arr,
        a_hist,b_hist,c_hist,d_hist,e_hist) = jax.lax.scan(_scan_training, (a0,b0,d0),next_keys)
    
    return [all_obs_arr,all_true_s_arr,all_u_arr,all_qs_arr,all_qs_post,all_qpi_arr,efes_arr,a_hist,b_hist,d_hist]

def synthetic_training_multi_subj(rngkeys_for_all_subjects,
            Ntrials,T,
            A,B,D,U,
            a0,b0,c,d0,e,u,
            planning_options = DEFAULT_PLANNING_OPTIONS,
            action_selection_options = DEFAULT_ACTION_SELECTION_OPTIONS,
            learning_options = DEFAULT_LEARNING_OPTIONS):

    map_this_function = partial(synthetic_training,
            Ntrials=Ntrials,T=T,
            A=A,B=B,D=D,U=U,
            a0=a0,b0=b0,c=c,d0=d0,e=e,u=u,
            planning_options = planning_options, 
            action_selection_options=action_selection_options,
            learning_options = learning_options)
    mapped_over_subjects = vmap(map_this_function)(rngkeys_for_all_subjects)
    return mapped_over_subjects


# Likelihood function (used for fitting !)
def empirical(obs_vect,act_vect,
        pa0,pb0,c,pd0,e,u,
        planning_options = DEFAULT_PLANNING_OPTIONS,
        learning_options = DEFAULT_LEARNING_OPTIONS):
    """,
    This method uses the compute_trial_posteriors_empirical function from the ai_jax_loop .py file
    It provides active inference agents with observation and returns their action posterior depending on their internal parameters.
    To allow for better convergence, the empirical actions at time t are observed at time t+1 instead of relying
    on computed action posteriors.
    
    Inputs : 
    - obs_vect : one_hot-encoded observations along a list of observation modalities. Each tensor in the list is of size Ntrials x Ntimesteps x Nobservations
    - act_vect : one_hot-encoded *observed* actions. This tensor is of size Ntrials x Ntimesteps x Nactions
    - pa0, pb0, c, pd0, e , U the agent model priors
    - learning & planning options for this agent (static). Note that action selection is not performed by this method.
    This method accounts for training wide effects by performing learning updates at the end of each trial.
    """
    
    def _scan_training(carry,data_trial):
        
        (pre_a,pre_b,pre_d) = carry
        
        (obs_trial,act_trial) = data_trial
        
        
        trial_a,trial_b,trial_d = vectorize_weights(pre_a,pre_b,pre_d,u)
        trial_c,trial_e = to_log_space(c,e)
        trial_a_nov,trial_b_nov = get_vectorized_novelty(pre_a,pre_b,u,compute_a_novelty=True,compute_b_novelty=True)
        
        # Empirical based state + action posterior for the whole trial
        qs_arr,qpi_arr = empirical_trial(obs_trial,act_trial,
                                trial_a,trial_b,trial_c,trial_d,trial_e,
                                trial_a_nov,trial_b_nov,
                                include_last_observation=True,
                                planning_options=planning_options)

        # NO ACTION SELECTION HERE ! We're using empirical observations 
        # to compute the evolution of model parameters instead
        
        # # Then, we update the parameters of our HMM model at this level
        # a_post,b_post,_,d_post,_,qs_post = learn_after_trial(obs_trial,qs_arr,act_trial,
        #                                          pre_a,pre_b,c,pre_d,e,u,
        #                                          learn_what=learning_options["bool"],
        #                                          learn_rates=learning_options["learning_rates"],
        #                                          forget_rates=learning_options["forgetting_rates"],
        #                                          post_trial_smooth=learning_options["smooth_states"])
        
        a_post,b_post,c_post,d_post,e_post,qs_post = learn_after_trial(obs_trial,qs_arr,act_trial,
                                            pre_a,pre_b,c,pre_d,e,u,
                                            method = learning_options["method"],
                                            learn_what = learning_options["bool"],
                                            learn_rates=learning_options["learning_rates"],
                                            forget_rates=learning_options["forgetting_rates"],
                                            generalize_state_function=learning_options["state_generalize_function"],
                                            generalize_action_table=learning_options["action_generalize_table"],
                                            cross_action_extrapolation_coeff=learning_options["cross_action_extrapolation_coeff"],
                                            em_iter = learning_options["em_iterations"])
        
        return (a_post,b_post,d_post),(qs_arr,qs_post,qpi_arr,a_post,b_post,d_post)
    
    final_matrices,(training_qs_arr,training_qs_post,training_qpi_arr,training_a_post,training_b_post,training_d_post) = jax.lax.scan(_scan_training,(pa0,pb0,pd0),(obs_vect,act_vect))
    
    return [training_qs_arr,training_qs_post,training_qpi_arr,training_a_post,training_b_post,training_d_post]