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
import tensorflow_probability.substrates.jax.distributions as tfd

from functools import partial
from fastprogress.fastprogress import progress_bar

from .jax_toolbox import _normalize,_jaxlog,_swapaxes,_condition_on

from .shape_tools import to_vec_space,to_source_space
from .shape_tools import to_vec_space_a,to_source_space_a
from .shape_tools import vectorize_weights

from .layer_infer_state import compute_state_posterior,_compute_log_likelihood_multi_mod

from .jax_toolbox import weighted_padded_roll


# _______________________________________________________________________________________
# ________________________________ Dirichlet Weights Updating ___________________________
# _______________________________________________________________________________________
def learn_a(hist_obs,hist_qs,pa,lr_a):
    def _learn_a_mod(o_mod,pa_mod):
        da = jnp.einsum("ti,tj->ijt",o_mod,hist_qs) # For flattened emission mappings only !
        return pa_mod + lr_a*da.sum(axis=-1)
    return tree_map(_learn_a_mod,hist_obs,pa)

def learn_b(hist_u,hist_qs,pb,lr_b):
    post_qs = hist_qs[1:,:]
    pre_qs = hist_qs[:-1,:]
    db = jnp.einsum("ti,tj,tu->ijut",post_qs,pre_qs,hist_u) 
                # For flattened transition mappings only !
    return pb + lr_b*db.sum(axis=-1)


def learn_b_generalize(hist_u,hist_qs,pb,lr_b,
                       generalize_function):
    post_qs = hist_qs[1:,:]
    pre_qs = hist_qs[:-1,:]
    db = jnp.einsum("ti,tj,tu->ijut",post_qs,pre_qs,hist_u) 
                # For flattened transition mappings only !
    
    # Across all timesteps :
    db_all_timesteps = db.sum(axis=-1)
    
    # db is the history of state transitions for each action and at each timestep
    # Making some broad hypotheses about the structure of the latent state space,
    # we can generalize the findings at a coordinate $(s_{t+1},s_t)$ to over states
    # This can be done by rolling the db_all_timesteps matrix across s_(t+1) and s_t axes
    # simultaneously : 
    gen_db = vmap(lambda bu : weighted_padded_roll(bu,generalize_function),in_axes=(-1))(db_all_timesteps)
    gen_db = jnp.moveaxis(gen_db,0,-1)
    
    return pb + lr_b*gen_db


def learn_b_factorized(hist_u_tree,hist_qs_tree,pb_tree,lr_b,
                    linear_state_space=False,generalize_fadeout_function=None):
    
    # For now, we assume a single generalize fading function across all factors
    def learn_b_factor(hist_u_factor,hist_qs_factor,b_factor,linear_state_space_f):
        if linear_state_space_f:
            assert generalize_fadeout_function != None, "State space action generalization requires a fadeout function."
            b = learn_b_generalize(hist_u_factor,hist_qs_factor,b_factor,lr_b,generalize_fadeout_function)
        else:
            b = learn_b(hist_u_factor,hist_qs_factor,b_factor,lr_b) 
        return b
    
    if (type(linear_state_space)==list):
        assert len(linear_state_space)==len(hist_u_tree),"If linear_state_space is a list, it should match the number of state factors"
        
        # assert type(generalize_fadeout_function)==list,"If state space structure assumption is a list, there should be a corr"
        function_to_map = learn_b_factor
        return tree_map(function_to_map,hist_u_tree,hist_qs_tree,pb_tree,linear_state_space)
    else :
        function_to_map = partial(learn_b_factor,linear_state_space_f=linear_state_space)
        return tree_map(function_to_map,hist_u_tree,hist_qs_tree,pb_tree)



def learn_d(hist_qs,pd,lr_d):
    # We only look at the first timestep :
    return pd + lr_d*hist_qs[0,:]

def learn_d_factorized(hist_qs_tree,pd_tree,lr_d):
    
    def learn_d_factor(hist_qs_factor,d_factor):
        return learn_d(hist_qs_factor,d_factor,lr_d)
    
    return tree_map(learn_d_factor,hist_qs_tree,pd_tree)


# State smoothing (move this somewhere else ?)
# Those are filters used a posteriori to refine posterior inference

def forwards_pass(hist_obs_vect,hist_u_vect,loc_a,loc_b,loc_d):
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
    # Compute the log likelihooods of each observations
    log_likelihoods_timestep_func = (lambda o_d_t : _compute_log_likelihood_multi_mod(loc_a, o_d_t))
    logliks = vmap(log_likelihoods_timestep_func)(hist_obs_vect)
    
    num_timesteps, num_states = logliks.shape

    def _step(carry, t):
        carry_log_norm,prior_this_timestep = carry
        
        forward_filt_probs, log_norm = _condition_on(prior_this_timestep, logliks[t])
        
        carry_log_norm = carry_log_norm + log_norm
        
        
        
        # Predict the next state (going forward in time).
        transition_matrix = jnp.einsum("iju,u->ij",loc_b,hist_u_vect[t-1,...])
        prior_next_timestep = transition_matrix @ forward_filt_probs
                
        return (carry_log_norm,prior_next_timestep),forward_filt_probs

    init_carry = (jnp.array([0.0]),loc_d)
    (log_norm,_),forward_pred_probs = lax.scan(_step, init_carry, jnp.arange(num_timesteps))
    
    return log_norm,forward_pred_probs
    
def backwards_pass(hist_obs_vect,hist_u_vect,loc_a,loc_b):
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
    # Compute the log likelihooods of each observations
    log_likelihoods_timestep_func = (lambda o_d_t : _compute_log_likelihood_multi_mod(loc_a, o_d_t))
    logliks = vmap(log_likelihoods_timestep_func)(hist_obs_vect)
    
    num_timesteps, num_states = logliks.shape

    def _step(carry, t):
        carry_log_norm,prior_this_timestep = carry
        
        backward_filt_probs, log_norm = _condition_on(prior_this_timestep, logliks[t])
        
        carry_log_norm = carry_log_norm + log_norm
        
        # Predict the next (previous) state (going backward in time).
        transition_matrix = jnp.einsum("iju,u->ij",loc_b,hist_u_vect[t-1,...])
        prior_previous_timestep = transition_matrix.T @ backward_filt_probs
                
        return (carry_log_norm,prior_previous_timestep),backward_filt_probs

    init_carry = (jnp.array([0.0]),jnp.ones(num_states))
    (log_norm,_),backward_pred_probs_rev = lax.scan(_step, init_carry, jnp.arange(num_timesteps)[::-1])
    
    backward_pred_probs = backward_pred_probs_rev[::-1]
    return log_norm, backward_pred_probs

def smooth_posterior_after_trial(hist_obs_vect,hist_qs,hist_u_vect,
          pa,pb,pd,U, filter_type="two_filter"):

    smoothed_posterior = hist_qs
    if filter_type=="two_filter":
        loc_a,loc_b,loc_d = vectorize_weights(pa,pb,pd,U)
        
        ll_for,forwards_smooths = forwards_pass(
                hist_obs_vect,hist_u_vect,
                loc_a,loc_b,loc_d
        )
        ll_back,backwards_smooths = backwards_pass(
                hist_obs_vect,hist_u_vect,
                loc_a,loc_b
        )
        smoothed_posterior,_ = _normalize(forwards_smooths*backwards_smooths,axis=-1)
        
    elif filter_type=="one_filter":
        loc_a,loc_b,loc_d = vectorize_weights(pa,pb,pd,U)
        
        forwards_smooths = hist_qs 
            # Assume that the forward filtering was done already during trial
        ll_back,backwards_smooths = backwards_pass(
                hist_obs_vect,hist_u_vect,
                loc_a,loc_b
        )
        smoothed_posterior,_ = _normalize(forwards_smooths*backwards_smooths,axis=-1)
        
    elif filter_type == "rts_direct":  #a discrete implementation of a 
                                        #  Rauch-Tung-Striebel smoother
        raise NotImplementedError("The developer is lazy , this has not been done yet :'(")
        
    return smoothed_posterior


# _______________________________________________________________________________________
# Playing with allowable actions : 
# switching from a vectorized space (all actions in one dimension)
# to a factorized space (allowable actions for each factor)
def vectorize_factorwise_allowable_actions(_u,_B):
    # This needs to be mapped across state factors ! 
    def factorwise_allowable_action_vectors(idx,B_f):
        return jax.nn.one_hot(idx,B_f.shape[-1])
    
    # This function takes one of the action index, and decomposes it into action vectors across all factors
    map_function = (lambda _x : tree_map(factorwise_allowable_action_vectors,list(_x),_B))
    
    return (vmap(map_function)(_u))
    
def posterior_transition_index_factor(transition_dict,posterior):
    def posterior_transition_factor(allowable_action_factor):
        return jnp.einsum("ij,ti->tj",allowable_action_factor,posterior)
    return tree_map(posterior_transition_factor,transition_dict)
# _______________________________________________________________________________________


# Main function : 
def learn_after_trial(hist_obs_vect,hist_qs,hist_u_vect,
          pa,pb,pc,pd,pe,U,
          learn_what={"a":True,"b":True,"c":False,"d":True,"e":False},
          learn_rates={"a":1.0,"b":1.0,"c":0.0,"d":1.0,"e":0.0},
          post_trial_smooth = True,
          assume_linear_state_space=False,generalize_fadeout_function=None):
        
    hist_qs_loc = hist_qs
    if post_trial_smooth:
        hist_qs_loc  = smooth_posterior_after_trial(hist_obs_vect,hist_qs,hist_u_vect,
                                pa,pb,pd,U,"one_filter")
    
    # learning a requires the states to be in vectorized mode
    a = pa
    if learn_what["a"]:
        # hist_obs_vect = _swapaxes(hist_obs_vect,tree=True)
        post_a = learn_a(hist_obs_vect,hist_qs_loc,to_vec_space_a(pa),learn_rates["a"])
        a = to_source_space_a(post_a,pa[0].shape[1:])
    
    
    # learning b and d requires the states to be in factorized mode : 
    source_space_shape = pa[0].shape[1:]  # This is the shape of the source space
    qs_hist_all_f = vmap(lambda x : to_source_space(x,source_space_shape))(hist_qs_loc)

    d = pd    
    if learn_what["d"]:
        d = learn_d_factorized(qs_hist_all_f,pd,learn_rates["d"])
        # d = learn_d(hist_qs_loc,pd,learn_rates["d"])
        
    b = pb
    if learn_what["b"]:
        # This is constant across trials, TODO : integrate into a class
        u_all = vectorize_factorwise_allowable_actions(U,pb)
        
        # This changes every trial : 
        u_hist_all_f = posterior_transition_index_factor(u_all,hist_u_vect)
        
        b = learn_b_factorized(u_hist_all_f,qs_hist_all_f,pb,learn_rates["b"],assume_linear_state_space,generalize_fadeout_function)
  
    c = pc
    if learn_what["c"]:
        raise NotImplementedError("TODO !") 
    
    e = pe
    if learn_what["e"]:
        raise NotImplementedError("TODO !") 
    
    return a,b,c,d,e,hist_qs_loc

if __name__ == "__main__":
    Ns = 10
    T = 100
    Np = 10

    Nos = np.array([10,8])
    fixed_observations = [np.random.randint(0,No,(T,)) for No in Nos]
    

    # A = [_normalize(jr.uniform(key,(No,Ns)))[0] for No in Nos]

    Nmod = 2
    key = jr.PRNGKey(464)   
    key,lockey = jr.split(key)
    a = [_normalize(jnp.eye(Ns))[0],_normalize(jr.uniform(lockey,(Nos[1],Ns)))[0]]

    fixed_observations = [np.random.randint(0,No,(T,)) for No in Nos]
    obs_vectors = [jax.nn.one_hot(rvs,No,axis=0) for rvs,No in zip(fixed_observations,Nos)]

    key,lockey = jr.split(key)
    qsm,_ = _normalize(jr.uniform(lockey,(Ns,T)))

    print(qsm)
    for obs in obs_vectors:
        print(obs)

    da = learn_a(obs_vectors,qsm,a)
    for obs in da:
        print(obs)

