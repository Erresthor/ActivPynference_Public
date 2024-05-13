import sys,os
import time
import numpy as np
import pickle 
import matplotlib.pyplot as plt

import actynf
from utils.basic_task import build_training_process,build_subject_model

import jax
from actynf.jax_methods.layer_training import training_step




# Our jax implementation has 2 major constraints w.r.t model definition :
# 1. Only one latent state dimension
# 2. No for loops (make the pipeline jax compatible) ! ;p

# Full synthetic trial
def effort_and_rest(full_trial_rng_key,
                    Ns,Nos,Np,T,
                    A_effort,a_effort,c_effort,
                    A_rest,a_rest,c_rest,
                    b,B,
                    d,D,
                    e,
                    alpha = 16,gamma = None):
    selection_method="stochastic"
    smooth_state_estimates=False
    
    # SYNTHETIC EFFORT STEP
    full_trial_rng_key,effort_rng_key = jax.random.split(full_trial_rng_key)
    learn_effort = {
        "bool":{"a":True,"b":True,"d":True},
        "rates":{"a":1.0,"b":1.0,"d":1.0}
    }
    effort_results = training_step(effort_rng_key,Ns,Nos,Np,T,
                        a_effort,b,d,c_effort,e,
                        A_effort,B,D,
                        Th =3,
                        selection_method=selection_method,alpha = alpha,gamma = gamma, 
                        learn_dictionnary = learn_effort,
                        smooth_state_estimates=smooth_state_estimates)
    
    (effort_obs_darr,effort_obs_arr,effort_obs_vect_arr,
        effort_true_s_darr,effort_true_s_arr,effort_true_s_vect_arr,
        effort_u_d_arr,effort_u_arr,effort_u_vect_arr,
        effort_qs_arr,effort_qpi_arr,effort_efes,
        effort_a_post,effort_b_post,effort_d_post) = effort_results
    b = effort_b_post
    d = effort_d_post
    
    full_trial_rng_key,rest_rng_key = jax.random.split(full_trial_rng_key)
    learn_rest = {
        "bool":{"a":False,"b":True,"d":True},
        "rates":{"a":1.0,"b":1.0,"d":1.0}
    }
    rest_results = training_step(rest_rng_key,Ns,Nos,Np,T,
                        a_rest,b,d,c_rest,e,
                        A_rest,B,D,
                        Th =3,
                        selection_method=selection_method,alpha = alpha,gamma = gamma, 
                        learn_dictionnary = learn_rest,
                        smooth_state_estimates=smooth_state_estimates)
    
    ( _,rest_obs_arr,_,
        rest_true_s_darr,rest_true_s_arr,rest_true_s_vect_arr,
        rest_u_d_arr,rest_u_arr,rest_u_vect_arr,
        rest_qs_arr,rest_qpi_arr,rest_efes,
        rest_a_post,rest_b_post,rest_d_post) = rest_results
    
    b = rest_b_post
    d = rest_d_post
    
    return effort_results,rest_results


# TODO for the inference : an empirical posterior model :)
# Action posterior given the last observation + parameter update based on observed actions