import os
import random as ra
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
import arviz as az
import corner  

from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.tree_util import tree_map
from jax import lax, vmap, jit
    
from .jax_toolbox import _normalize,_jaxlog,convert_to_one_hot_list
from .planning_tools import compute_novelty
from .layer_options import DEFAULT_PLANNING_OPTIONS,DEFAULT_ACTION_SELECTION_OPTIONS

from .layer_process import initial_state_and_obs,process_update,fetch_outcome
from .layer_infer_state import compute_state_posterior

from .layer_plan_classic import policy_posterior as policy_posterior_classic
from .layer_plan_sophisticated import policy_posterior as policy_posterior_sophisticated

from .layer_pick_action import sample_action

EOT_FILTER_CST = 2
    # How many more planned trials are in the filter (in excess of Th) 
    # (Goal : avoid the planner looping back
    # and finding 1.0 instead of 0.0 when overflowing)

# Compute the posteriors for each timestep during the trial : 
def compute_step_posteriors(t,prior,observation,
                            a,b,c,e,
                            a_novel,b_novel,
                            Th,filter_end_of_trial,
                            planning_options=DEFAULT_PLANNING_OPTIONS):   
    # State inference for the current timestep
    qs,F = compute_state_posterior(prior,observation,a)
            
    
    if planning_options["method"]=="sophisticated":
        # filter_end_of_trial = filter_end_of_trial[1:]
        efe,raw_qpi = policy_posterior_sophisticated(t,Th,filter_end_of_trial,
                                            qs,
                                            a,b,c,e,
                                            a_novel,b_novel,
                                            planning_options)
    elif planning_options["method"]=="classic":
        filter_end_of_trial = filter_end_of_trial[:-EOT_FILTER_CST]
        # Policy planning
        efe,raw_qpi = policy_posterior_classic(t,Th,filter_end_of_trial,
                                             qs,
                                             a,b,c,e,
                                             a_novel,b_novel,
                                             planning_options)
    elif planning_options["method"]=="gradient":
        filter_end_of_trial = filter_end_of_trial[:-EOT_FILTER_CST] # ?
        # Policy planning
        efe,raw_qpi = policy_posterior_classic(t,Th,filter_end_of_trial,
                                             qs,
                                             a,b,c,e,
                                             a_novel,b_novel,
                                             planning_options)
    else : 
        raise NotImplementedError("Not implemented planning method : " + str(planning_options["method"])) 
    return qs,raw_qpi,efe

# Ensure that timesteps outside the trial range are ignored :
def get_filter_eot(T,Th):
    """_summary_ : returns T-1 filters returning 1.0 if we are within the trial horizon, and 0.0 if we are not.

    Args:
        T (_type_): Trial duration (timestep)
        Th (_type_): Temporal horizon

    Returns:
        _type_: a tensor array [T-1 , Th + V] of filters with 1.0 where the trial is still ongoing 
                and 0.0 where its computations should not be performed.
                Being  at the end of the trial means that the EFE computations should return G(tau) = E.
    """     
    end_of_trial_scale = jnp.arange(T-1,-1,-1)  # [ T-1, T-2, ..., 0]
        # Timesteps within the trial (positives are within the trial time, negatives are ignored):
    
    def filter_for(k):
            # -1 because we're not accounting for the present state 
        return jax.nn.one_hot(end_of_trial_scale - k - 1,Th+EOT_FILTER_CST).sum(axis=-2)
    return vmap(filter_for)(jnp.arange(0,T-1,1))


# Run the actual trials :
def synthetic_trial(rngkey,T,
              A,B,D,
              a_norm,b_norm,c,d_norm,e,
              a_novel,b_novel,
              planning_options=DEFAULT_PLANNING_OPTIONS,
              action_selection_options=DEFAULT_ACTION_SELECTION_OPTIONS):
    """Computes the infered states and actions for T timesteps for a Sophisticated Inference agent.
    Agent options are written in planning options.
    All input weights must be vectorized (only 1 latent dim).

    Args:
        rngkey (_type_): _description_
        T (_type_): _description_
        Th (_type_): _description_
        A (_type_): _description_
        B (_type_): _description_
        D (_type_): _description_
        a_norm (_type_): _description_
        b_norm (_type_): _description_
        c (_type_): _description_
        d_norm (_type_): _description_
        e (_type_): _description_
        a_novel (_type_): _description_
        b_novel (_type_): _description_
        alpha (int, optional): _description_. Defaults to 16.
        selection_method (str, optional): _description_. Defaults to "stochastic".
        planning_options (_type_, optional): _description_. Defaults to DEFAULT_PLANNING_OPTIONS.

    Returns:
        _type_: _description_
    """
    rngkey, init_key = jr.split(rngkey)
    
    Th = planning_options["horizon"]
    
    # Initialize process
    [s_0_d,s_0_idx,s_0_vect],[o_0_d,o_0_idx,o_0_vect] = initial_state_and_obs(init_key,A,D)
    
    # Initialize subject model
    ps_0 = d_norm
    
    def _scan(carry,xs): # This describes posterior computation based on observations
                         # a action selection at timestep t.
                         # AND process update in the next timestep based on this action (t+1)
                         # It returns the process and model statistics
        (key,t,filter_end_of_trial) = xs
        
        key,key_agent,key_process = jr.split(key,3)  # For random generations
        
        # Saved states from previous process tick and model update (t-1) --------
        true_s,observation,prior = carry

        # ---------------------------------------------------------------------------------
        # Model update (t) ----------------------------------------------------------------
        
        # State & policy inference
        # jax.debug.print("observations: {}", observation)
        qs,raw_qpi,efe = compute_step_posteriors(t,prior,observation,a_norm,b_norm,c,e,a_novel,b_novel,Th,filter_end_of_trial,planning_options)
        # jax.debug.print("efe: {}", efe)
        
        # Action sampling
        u_d,u_idx,u_vect = sample_action(raw_qpi,action_selection_options,rng_key=key_agent)
        
        # Prior for next timestep
        new_prior = jnp.einsum("iju,j,u->i",b_norm,qs,u_vect)

        # ---------------------------------------------------------------------------------
        # Process update (t+1) -------------------------------------------------- 
        [s_d,s_idx,s_vect],[o_d,o_idx,o_vect] = process_update(key_process,true_s,A,B,u_vect)
        
        return (s_vect,o_vect,new_prior),(o_d,o_idx,o_vect,s_d,s_idx,s_vect,u_d,u_idx,u_vect,qs,raw_qpi,efe)
    
    timestamps = jnp.arange(T-1)  # 0 to t-1
    end_of_trial_filters = get_filter_eot(T,Th)  
            # End Of Trial filter : vectors of 1.0 or 0.0 depending on if we reached T-1
    next_keys = jr.split(rngkey, T - 1)
    (last_true_s,last_obs,last_prior), (obs_darr,obs_arr,obs_vect_arr,true_s_darr,true_s_arr,true_s_vect_arr,u_d_arr,u_arr,u_vect_arr,qs_arr,qpi_arr,efes) = jax.lax.scan(_scan, (s_0_vect,o_0_vect,ps_0),(next_keys,timestamps,end_of_trial_filters))
    
    
    # Compute the state posterior for the ultimate timestep
    last_qs,_ = compute_state_posterior(last_prior,last_obs,a_norm)

    # Concatenate first and last elements. 
    # TODO : A derivative of this function doing everything in the scan function to avoid this kind of operations
    # 1. Observations
    obs_darr = tree_map(lambda x,y :jnp.concatenate([x.reshape(1,-1),y],axis=0),o_0_d, obs_darr)
    obs_arr = tree_map(lambda x,y :jnp.concatenate([jnp.expand_dims(x,axis=0),y],axis=0),o_0_idx, obs_arr)
    obs_vect_arr = tree_map(lambda x,y :jnp.concatenate([x.reshape(1,-1),y],axis=0),o_0_vect, obs_vect_arr)
    # 1. Process states     
    true_s_darr = jnp.concatenate([s_0_d.reshape(1,-1),true_s_darr],axis=0)
    true_s_arr = jnp.concatenate([jnp.expand_dims(s_0_idx,axis=0),true_s_arr],axis=0)
    true_s_vect_arr = jnp.concatenate([s_0_vect.reshape(1,-1),true_s_vect_arr],axis=0)
    # 2. Model state inferences   
    qs_arr = jnp.concatenate([qs_arr,last_qs.reshape(1,-1)],axis=0)

    return [obs_darr,obs_arr,obs_vect_arr,true_s_darr,true_s_arr,true_s_vect_arr,u_d_arr,u_arr,u_vect_arr,qs_arr,qpi_arr,efes]

# An old function to use set values for process states & observations
def _depr_synthetic_trial_set_vals(rngkey,T,
              pa,pb,c,pd,e,
              A,B,D,
              static_set_states=None,static_set_obs=None,
              alpha = 16, selection_method="stochastic",
              planning_options=DEFAULT_PLANNING_OPTIONS):
    """
    This method is similar to synthetic_trial, but it does not use a scan function, in order to allow
    for manual checking. It should also be much slower.

    Args:
        rngkey (_type_): _description_
        Ns (_type_): _description_
        Nos (_type_): _description_
        Np (_type_): _description_
        pa (_type_): _description_
        pb (_type_): _description_
        c (_type_): _description_
        pd (_type_): _description_
        e (_type_): _description_
        A (_type_): _description_
        B (_type_): _description_
        D (_type_): _description_
        static_set_states (_type_, optional): _description_. Defaults to None.
        static_set_obs (_type_, optional): _description_. Defaults to None.
        T (int, optional): _description_. Defaults to 10.
        Th (int, optional): _description_. Defaults to 3.
        alpha (int, optional): _description_. Defaults to 16.
        gamma (_type_, optional): _description_. Defaults to None.
        selection_method (str, optional): _description_. Defaults to "stochastic".
        planning_options (_type_, optional): _description_. Defaults to DEFAULT_PLANNING_OPTIONS.

    Returns:
        _type_: _description_
    """
    
    
    # Model dirichlet parameter vectors : (this should probably done by the parent class ?)
    
    # 1. Normalize the subject priors ( = get their expected values 
    # given the entertained dirichlet prior)
    a = _normalize(pa,tree=True)
    b,_ = _normalize(pb)
    d,_ = _normalize(pd)

    # Compute the prior novelty, if we need it : 
    if planning_options["a_novelty"]:
        a_novel = compute_novelty(pa,True)
    else : 
        a_novel = None
        
    if planning_options["b_novelty"]:
        b_novel = compute_novelty(pb)
    else : 
        b_novel = None
    
    rngkey, init_key = jr.split(rngkey)


    [s_0_d,s_0_idx,s_0_vect],[o_0_d,o_0_idx,o_0_vect] = fetch_outcome(init_key,Ns,Nos,
            0,None,None,
            A,B,D,
            fixed_states_array=static_set_states,fixed_outcomes_tree=static_set_obs)
    
    # Initialize subject model
    ps_0 = d
    
    true_s,observation,prior = s_0_vect,o_0_vect,ps_0

    timestamps = jnp.arange(T-1)  # 0 to t-1
    end_of_trial_filters = end_of_trial_filter(T,Th)  # vectors of 1.0 or 0.0 depending on how close we are to T-1
    next_keys = jr.split(rngkey, T - 1)
    
    qss = []
    qpis = []
    efes = []
    
    true_s_d = [s_0_d]
    true_s_idx = [s_0_idx]

    true_o_d = [o_0_d]
    true_o_idx = [o_0_idx]
    for key,t,filter_end_of_trial in zip(next_keys,timestamps,end_of_trial_filters):
        key,key_agent,key_process = jr.split(key,3)  # For random generations      
        
        # ---------------------------------------------------------------------------------
        # Model update (t) ----------------------------------------------------------------
        
        # State & policy inference
        qs,raw_qpi,efe = compute_step_posteriors(t,prior,observation,a,b,c,e,a_novel,b_novel,gamma,Np,Th,filter_end_of_trial,planning_options)
        
        # Action sampling
        u_d,u_idx,u_vect = sample_action(raw_qpi,Np,alpha, selection_method=selection_method,rng_key=key_agent)

        # Prior for next timestep
        new_prior = jnp.einsum("iju,j,u->i",b,qs,u_vect)

        # ---------------------------------------------------------------------------------
        # Process update (t+1) -------------------------------------------------- 
        # [s_d,s_idx,s_vect],[o_d,o_idx,o_vect] = process_update(key_process,true_s,A,B,u_vect,Ns,Nos)
        
        [s_d,s_idx,s_vect],[o_d,o_idx,o_vect] = fetch_outcome(key_process,Ns,Nos,
                t+1,true_s,u_vect,
                A,B,D,
                fixed_states_array=static_set_states,fixed_outcomes_tree=static_set_obs)
         
        true_s,observation,prior = s_vect,o_vect,new_prior

        qss.append(qs)
        qpis.append(raw_qpi)
        efes.append(efe)
        
        true_s_idx.append(s_idx)
        true_s_d.append(s_d)
        
        true_o_idx.append(o_idx)
        true_o_d.append(o_d)
    
    # Compute the state posterior for the ultimate timestep
    last_qs,_ = compute_state_posterior(prior,observation,a)
    qss.append(last_qs)
    

    # return [obs_darr,obs_arr,obs_vect_arr,true_s_darr,true_s_arr,true_s_vect_arr,u_d_arr,u_arr,u_vect_arr,qs_arr,qpi_arr,efes]

    true_states = [jnp.stack(true_s_idx),jnp.stack(true_s_d)]
        
    true_o_d_transposed = list(zip(*true_o_d)) # true_o_d# list(map(list, zip(*true_o_d)))   
    true_o_d = []
    for modality,true_o_d_m in enumerate(true_o_d_transposed):
        true_o_d.append(jnp.array(true_o_d_m))
    true_obs = [jnp.array(true_o_idx),true_o_d]
    
    return jnp.stack(qss),jnp.stack(qpis),jnp.stack(efes),true_states,true_obs

# A prefilled partial for parrallelization, modify this depending on you needs
def _partial_synthetic_trial(key,T,
                vecA,vecB,vecD,
                veca,vecb,vecc,vecd,vece,
                nov_a,nov_b,
                selection_method,alpha,
                planning_options):
    parrallel_synth_trial = partial(synthetic_trial,                     
        rngkey = key,
        T=T,
        A=vecA,B=vecB,D=vecD,
        a_norm=veca,b_norm=vecb,c=vecc,d_norm=vecd,e=vece,
        a_novel=nov_a,b_novel=nov_b,
        selection_method=selection_method,alpha = alpha, 
        planning_options=planning_options)
    return parrallel_synth_trial

# STATE AND ACTION POSTERIOR IN RESPONSE TO EMPIRICAL OBSERVATION(S) + PREVIOUS ACTION(S)
# Get model likelihood for observed actions based on empirical observations  :
def empirical(obs_vect,act_vect,
              a_norm,b_norm,c,d_norm,e,
              a_novel,b_novel,
              include_last_observation=False,
              planning_options=DEFAULT_PLANNING_OPTIONS):
    T = act_vect.shape[0]+1
    Th = planning_options["horizon"]
    
    def _scan(carry,data_t):
        emp_prior = carry
        (observation_t,observed_action_t_vect,filter_t,t) = data_t
        
        qs,raw_qpi,efe = compute_step_posteriors(t,emp_prior,observation_t,
                                                 a_norm,b_norm,c,e,
                                                 a_novel,b_novel,
                                                 Th,filter_t,
                                                 planning_options)
        
        # action_t_vect = action_t
        next_emp_prior = jnp.einsum("iju,j,u->i",b_norm,qs,observed_action_t_vect)

        return next_emp_prior,(qs,raw_qpi)
    
    initial_prior = d_norm
    
    timestamps = jnp.arange(T-1)
    end_of_trial_filters = get_filter_eot(T,Th)
    all_obs_but_last = tree_map(lambda x : x[:-1,...],obs_vect)
    last_prior,(qs_arr,qpi_arr) = jax.lax.scan(_scan,initial_prior,(all_obs_but_last,act_vect,end_of_trial_filters,timestamps))
    
    if include_last_observation : # Useful if we want the subject's weights to evolve accross trials
        last_obs = tree_map(lambda x : x[-1,...],obs_vect)
        last_posterior,_ = compute_state_posterior(last_prior,last_obs,a_norm)
        qs_arr = jnp.concatenate([qs_arr,last_posterior.reshape(1,-1)],axis=0)
    
    return qs_arr,qpi_arr