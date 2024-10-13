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

from .jax_toolbox import _normalize
from .shape_tools import vectorize_weights

from .learning.parameter_updating import get_parameter_update
from .learning.smoothing import smooth_trial


def update_prior(prior,delta,lr=1.0,learn_bool=True):
    
    def _update_function_element(_prior_k,_delta_k,_lr_k,_bool_k):
        if _bool_k :
            return _prior_k+_lr_k*_delta_k
        else :
            return _prior_k
    
    if type(prior)==list:
        if type(learn_bool) != list:
            learn_bool = [learn_bool for k in prior]
        
        if type(lr) != list :
            lr = [lr for k in prior]
        
        return tree_map(_update_function_element,prior,delta,lr,learn_bool)
    else :
        return _update_function_element(prior,delta,lr,learn_bool)
        

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




def EM_jax_one_trial(vec_emission_hist,emission_bool_hist,
           vec_action_hist,action_bool_hist,
           true_a_prior,true_b_prior,true_d_prior,U,
           N_iterations = 16,
           learn_what={"a":True,"b":True,"c":False,"d":True,"e":False},
           learn_rates={"a":1.0,"b":1.0,"c":0.0,"d":1.0,"e":0.0},
           state_generalize_function=None,action_generalize_table=None,cross_action_extrapolation_coeff=0.1):
    """EM algorithm for a HMM using a history of observations and actions for a single trial.

    Args:
        vec_emission_hist (_type_): _description_
        emission_bool_hist (_type_): _description_
        vec_action_hist (_type_): _description_
        action_bool_hist (_type_): _description_
        true_a_prior (_type_): _description_
        true_b_prior (_type_): _description_
        true_d_prior (_type_): _description_
        U (_type_): _description_
        N_iterations (int, optional): _description_. Defaults to 16.
        state_generalize_function (_type_, optional): _description_. Defaults to None.
        action_generalize_table (_type_, optional): _description_. Defaults to None.
        cross_action_extrapolation_coeff (float, optional): _description_. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    
    # Checks :
    for mod in range(len(vec_emission_hist)):
        assert vec_emission_hist[mod].ndim==2, "Observations should have 2 dimensions : Ntimesteps x Noutcomes but has " + str(vec_emission_hist[mod].ndim) + " for modality " + str(mod)
    assert vec_action_hist.ndim==2, "Observed actions should have 2 dimensions : (Ntimesteps-1) x Nu"
    assert action_bool_hist.shape == vec_action_hist.shape[:-1], "The action filter should be of shape (Ntimesteps-1)"
    
    assert U.ndim == 2,"U should be a 2 dimensionnal mapping between action and transitions but is a {} dimensional tensor".format(U.ndim)
    
    
    # Static shapes __________________________________________________________________
    Nf = len(true_b_prior)
    Nu,Nf2 = U.shape                                
            # How many allowable actions there are
            # Each allowable action results in a specific transition for each factor
    
    assert Nf2==Nf,"Mismatch in the number of state factors. Please check the function inputs."
                                                                             
    Ns = true_a_prior[0].shape[1:] # This is the shape of the hidden state space (fixed for this model)
    
    
    
    # Let's get a factorized version of the action history : 
    vec_transition_per_factor = vectorize_factorwise_allowable_actions(U,true_b_prior)
    vec_transition_history = posterior_transition_index_factor(vec_transition_per_factor,vec_action_hist)
                        # A list of transitions performed per factor !
                        # across Ntrials x (Ntimesteps-1)
                        # y axis : states(t+1), x axis : states (t)
    # ___________________________________________________________________________________
    
    
    
    # Useful functions :  
    get_parameter_update_iteration = partial(get_parameter_update,Ns=Ns,Nu = Nu,
                                   state_generalize_function = state_generalize_function,
                                   action_generalize_table=action_generalize_table,cross_action_extrapolation_coeff=cross_action_extrapolation_coeff)
    
    # The actual EM : for N iterations
    # we will alternate hidden state estimation and parameter updates :
    def _scanner(carry,xs):
        # These are the parameters for this e-step iteration
        _it_a, _it_b,_it_d = carry
        
        _it_vec_a,_it_vec_b,_it_vec_d = vectorize_weights(_it_a,_it_b,_it_d,U)
        smoothed_posteriors_it,(ll_trials_it,ll_trials_hist_it) = smooth_trial(vec_emission_hist,vec_action_hist,emission_bool_hist,
                                                                            _it_vec_a,_it_vec_b,_it_vec_d,
                                                                            "two_filter",None)
            # A Ntrials x Ntimesteps x Ns tensor of smoothed state posteriors !

        delta_a,delta_b,_,delta_d,_ = get_parameter_update_iteration(vec_emission_hist,vec_transition_history,vec_action_hist,
                                                            emission_bool_hist,action_bool_hist,
                                                            smoothed_posteriors_it)
        
        _new_a = update_prior(true_a_prior,delta_a,learn_rates["a"],learn_what["a"])
        _new_b = update_prior(true_b_prior,delta_b,learn_rates["b"],learn_what["b"])
        _new_d = update_prior(true_d_prior,delta_d,learn_rates["d"],learn_what["d"])
        
        return (_new_a,_new_b,_new_d),(smoothed_posteriors_it,ll_trials_it)

    init_carry = (true_a_prior,true_b_prior,true_d_prior)
    (final_a,final_b,final_d),(smoothed_states,elbo_history) = lax.scan(_scanner, init_carry, jnp.arange(N_iterations))
    
    # # Last smoothed posterior : 
    final_smoothed_posteriors = smoothed_states[-1,...]
    # vec_final_a,vec_final_b,vec_final_d = vectorize_weights(final_a,final_b,final_d,U)
    # final_smoothed_posteriors,(final_elbo,final_ll_hist) = smooth_trial(vec_emission_hist,vec_action_hist,emission_bool_hist,
    #                                         vec_final_a,vec_final_b,vec_final_d)
    
    
    return (final_a,final_b,final_d),final_smoothed_posteriors,elbo_history


# Main function : 
def learn_after_trial(hist_obs_vect,hist_qs,hist_u_vect,
          pa,pb,pc,pd,pe,U,
          method="vanilla",
          learn_what={"a":True,"b":True,"c":False,"d":True,"e":False},
          learn_rates={"a":1.0,"b":1.0,"c":0.0,"d":1.0,"e":0.0},
          generalize_state_function=None,generalize_action_table=None,
          cross_action_extrapolation_coeff=0.1,em_iter = 4): 
       
    Nu,Nf = U.shape                                
    Ns = pa[0].shape[1:] # This is the shape of the hidden state space (fixed for this model)
    
    
    # This is constant across trials (For each action, what is the encoded factor transition)
    # TODO : integrate into a class
    u_all = vectorize_factorwise_allowable_actions(U,pb)
    
    # This changes every trial : actual history of encoded factor transition 
    u_hist_all_f = posterior_transition_index_factor(u_all,hist_u_vect)
    
    
    
    
    
    if method =="em":    
        
        # Assuming that we saw all actions & emissions :
        emission_bool_hist = [jnp.ones(o_d.shape[:-1]) for o_d in hist_obs_vect]
        action_bool_hist = jnp.ones_like(hist_u_vect[:,0])
            
        (final_a,final_b,final_d),final_smoothed_posteriors,elbo_history= EM_jax_one_trial(hist_obs_vect,emission_bool_hist,
                                        hist_u_vect,action_bool_hist,
                                        pa,pb,pd,U,
                                        N_iterations = em_iter,
                                        learn_what=learn_what,
                                        learn_rates=learn_rates,
                                        state_generalize_function=generalize_state_function,action_generalize_table=generalize_action_table,
                                        cross_action_extrapolation_coeff=cross_action_extrapolation_coeff)
        
        return final_a,final_b,pc,final_d,pe,final_smoothed_posteriors
    
    elif "vanilla" in method :
        
        # Assuming that we saw all actions & emissions :
        emission_bool_hist = [jnp.ones(o_d.shape[:-1]) for o_d in hist_obs_vect]
        action_bool_hist = jnp.ones_like(hist_u_vect[:,0])
        
        
        hist_qs_loc = hist_qs
        if method=="vanilla+backwards":
            _it_vec_a,_it_vec_b,_it_vec_d = vectorize_weights(pa,pb,pd,U)
            hist_qs_loc,_  = smooth_trial(hist_obs_vect,hist_u_vect,None,
                                    _it_vec_a,_it_vec_b,_it_vec_d,
                                    "one_filter",hist_qs)
        
        # learning a requires the states to be in vectorized mode
        da,db,dc,dd,de = get_parameter_update(hist_obs_vect,u_hist_all_f,hist_u_vect,
                                emission_bool_hist,action_bool_hist,
                                hist_qs_loc,
                                Ns,Nu,
                                state_generalize_function=generalize_state_function,
                                action_generalize_table=generalize_action_table,
                                cross_action_extrapolation_coeff=cross_action_extrapolation_coeff)
        
        
        final_a = update_prior(pa,da,learn_rates["a"],learn_what["a"])
        final_b = update_prior(pb,db,learn_rates["b"],learn_what["b"])
        final_d = update_prior(pd,dd,learn_rates["d"],learn_what["d"])
        final_e = update_prior(pe,de,learn_rates["e"],learn_what["e"])
        
        return final_a,final_b,pc,final_d,final_e,hist_qs_loc
        
    else : 
        raise NotImplementedError("Learning method {} has not been implemented.".format(method))
    
        
    
    

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