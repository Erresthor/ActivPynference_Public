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

from functools import partial

from actynf.jaxtynf.shape_tools import vectorize_weights

from .smoothing import smooth_trial,smooth_trial_window
from .parameter_updating import get_parameter_update


# _______________________________________________________________________________________
# Playing with allowable actions : 
# switching from a vectorized space (all actions in one dimension)
# to a factorized space (allowable actions for each factor)
def vectorize_factorwise_allowable_actions(_u,_Nactions):
    """ 
    From an array of indices of actions
     - /!\,_u should be 2 dimensionnal (Nfactors x Nactions)
    To a list of arrays of size (N_allowable_actions x Npossible_transitions_for_factor)
    """
    
    assert _u.ndim == 2,"_u should be a 2 dimensionnal mapping between action and transitions but is a {} dimensional tensor".format(_u.ndim)
    
    # This needs to be mapped across state factors ! 
    def factorwise_allowable_action_vectors(idx,N_actions_f):
        return jax.nn.one_hot(idx,N_actions_f)
    
    # This function takes one of the action index, and decomposes it into action vectors across all factors
    map_function = (lambda _x : tree_map(factorwise_allowable_action_vectors,list(_x),_Nactions))
    
    return (vmap(map_function)(_u))
    
def posterior_transition_index_factor(transition_list,history_of_actions):
    def posterior_transition_factor(allowable_action_factor):
        return jnp.einsum("ij,kti->ktj",allowable_action_factor,history_of_actions)
    return tree_map(posterior_transition_factor,transition_list)




# Main issue with this algorithm : compress multiple state factors. 
# Sophisticated inference chooses to compute belief propagation using a single latent dimension.
# Thus, we use a kronecker product to "flatten" multiple state dimensions, when they exist

# This is a problem when comes the time to learn transition matrix, especially when the model is equipped
# with some generalization ability along specific factors.
# We need to flatten the B and D matrices for the E-step, 
# and then marginalize the resulting state posteriors for the M-step
@partial(jit,static_argnames=["N_iterations","is_learn_a","is_learn_b","is_learn_d","state_generalize_function"])
def EM_jax(vec_emission_hist,emission_bool_hist,
           vec_action_hist,action_bool_hist,
           true_a_prior,true_b_prior,true_d_prior,U,
           N_iterations = 16,
           lr_a=1.0,lr_b=1.0,lr_d=1.0 ,
           is_learn_a = True,is_learn_b = True,is_learn_d = True,
           state_generalize_function=None,action_generalize_table=None,cross_action_extrapolation_coeff=0.1):
    """EM algorithm for a HMM using a history of observations and actions across multiple trials.

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
        lr_a (float, optional): _description_. Defaults to 1.0.
        lr_b (float, optional): _description_. Defaults to 1.0.
        lr_d (float, optional): _description_. Defaults to 1.0.
        is_learn_a (bool, optional): _description_. Defaults to True.
        is_learn_b (bool, optional): _description_. Defaults to True.
        is_learn_d (bool, optional): _description_. Defaults to True.
        state_generalize_function (_type_, optional): _description_. Defaults to None.
        action_generalize_table (_type_, optional): _description_. Defaults to None.
        cross_action_extrapolation_coeff (float, optional): _description_. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    
    
    # Checks :
    for mod in range(len(vec_emission_hist)):
        assert vec_emission_hist[mod].ndim==3, "Observations should have 3 dimensions : Ntrials x Ntimesteps x Noutcomes but has " + str(vec_emission_hist[mod].ndim) + " for modality " + str(mod)
    assert vec_action_hist.ndim==3, "Observed actions should have 3 dimensions : Ntrials x (Ntimesteps-1) x Nu"
    assert action_bool_hist.shape == vec_action_hist.shape[:-1], "The action filter should be of shape Ntrials x (Ntimesteps-1)"
    
    assert U.ndim == 2,"U should be a 2 dimensionnal mapping between action and transitions but is a {} dimensional tensor".format(U.ndim)
    
    
    # Static shapes __________________________________________________________________
    Nmod = len(true_a_prior)
    Nf = len(true_b_prior)
    Nu,Nf2 = U.shape                                
            # How many allowable actions there are
            # Each allowable action results in a specific transition for each factor
    
    assert Nf2==Nf,"Mismatch in the number of state factors. Please check the function inputs."
                                                                             
    Nuf = [b_f.shape[-1] for b_f in true_b_prior]   # How many transitions are possible per factor
    Ns = true_a_prior[0].shape[1:] # This is the shape of the hidden state space (fixed for this model)
    
    # Normalize the options to match the factors / modalities
    # (a.k.a all parameters should be lists of objects!)
    def _norm_option(_field,_list_shape):
        if type(_field) != list:
            _field_value = _field # Unecessary step ?
            _field = [_field_value for k in range(_list_shape)]
        assert len(_field)==_list_shape, "Length of field " + str(_field) + " does not match the required dimension " + str(_list_shape)
        return _field
    lr_a, is_learn_a = _norm_option(lr_a,Nmod),_norm_option(is_learn_a,Nmod)
    lr_b, is_learn_b = _norm_option(lr_b,Nf),_norm_option(is_learn_b,Nf)
    lr_d, is_learn_d = _norm_option(lr_d,Nf),_norm_option(is_learn_d,Nf)
    
    # Let's get a factorized version of the action history : 
    vec_transition_per_factor = vectorize_factorwise_allowable_actions(U,Nuf)
    vec_transition_history = posterior_transition_index_factor(vec_transition_per_factor,vec_action_hist)
                        # A list of transitions performed per factor !
                        # across Ntrials x (Ntimesteps-1)
                        # y axis : states(t+1), x axis : states (t)
    # ___________________________________________________________________________________
    
    
    
    # Useful functions :  
    get_param_variations = partial(get_parameter_update,Ns=Ns,Nu = Nu,
                                   state_generalize_function = state_generalize_function,
                                   action_generalize_table=action_generalize_table,cross_action_extrapolation_coeff=cross_action_extrapolation_coeff)
    
    def _update_prior(_prior,_param_variation,_lr=1.0,_learn_bool=True):
        if _learn_bool:
            # Here, we sum across all past trials. We could implement a
            # version where further trials impact parameter updates less !
            # (some kind of memory loss)
            return _prior + _lr*_param_variation.sum(axis=0)
        return _prior
    
    # The actual EM : for N iterations / until convergence 
    # we will alternate hidden state estimation
    # and parameter updates :
    def _scanner(carry,xs):
        # These are the parameters for this e-step iteration
        _it_a, _it_b,_it_d = carry
        
        _it_vec_a,_it_vec_b,_it_vec_d = vectorize_weights(_it_a,_it_b,_it_d,U)
        smoothed_posteriors,(ll_trials_it,ll_trials_hist_it) = smooth_trial_window(vec_emission_hist,vec_action_hist,emission_bool_hist,
                                                _it_vec_a,_it_vec_b,_it_vec_d)
            # A Ntrials x Ntimesteps x Ns tensor of smoothed state posteriors !
        
        # TODO : add log prob w.r.t. parameter priors to this !
        # see dynamax fit_em : 
        # lp = self.log_prior(params) + lls.sum()
        # Where for categorical distributions, the conjugate prior is the dirichlet distribution :
        # def log_prior(self, params):
        #      return tfd.Dirichlet(self.emission_prior_concentration).log_prob(params.probs).sum()
        # This should be quite similar to how we compute the EFE with learnable dynamics !
        delta_a,delta_b,_,delta_d,_ = vmap(get_param_variations)(vec_emission_hist,vec_transition_history,
                                                                 emission_bool_hist,action_bool_hist,
                                                                 smoothed_posteriors)
        
        _new_a = tree_map(_update_prior,true_a_prior,delta_a,lr_a,is_learn_a)
        _new_b = tree_map(_update_prior,true_b_prior,delta_b,lr_b,is_learn_b)
        _new_d = tree_map(_update_prior,true_d_prior,delta_d,lr_d,is_learn_d)
        
        return (_new_a,_new_b,_new_d),(smoothed_posteriors,ll_trials_it)

    init_carry = (true_a_prior,true_b_prior,true_d_prior)
    (final_a,final_b,final_d),(smoothed_states,elbo_history) = lax.scan(_scanner, init_carry, jnp.arange(N_iterations))
    
    # Last smoothed posterior : 
    vec_final_a,vec_final_b,vec_final_d = vectorize_weights(final_a,final_b,final_d,U)
    final_smoothed_posteriors,(final_elbo,final_ll_hist) = smooth_trial_window(vec_emission_hist,vec_action_hist,emission_bool_hist,
                                            vec_final_a,vec_final_b,vec_final_d)
    
    return (final_a,final_b,final_d),final_smoothed_posteriors,jnp.concatenate([elbo_history,jnp.expand_dims(final_elbo,-3)],axis=-3)


if __name__=="__main__":
    # Example of use :
    Ns_all = [2,3]
    Ntrials = 2
    Ntimesteps = 3
    
    # Transitions : 2 factors
    transitions = [np.zeros((Ns,Ns,Ns)) for Ns in Ns_all]
    for f,b_f in enumerate(transitions):
        for action in range(b_f.shape[-1]):
            
            b_f[...,action] = np.eye(Ns_all[f])
            try :
                b_f[action+1,action,action] += 1.0
            except :
                b_f[0,action,action] += 1.0
    raw_b = [jnp.array(b_f) for b_f in transitions]
    
    
    raw_d = [jnp.array([0.5,0.5]),jnp.array([0.5,0.5,0.0])]
    
    
    
    raw_a = [np.zeros((2,2,3)),np.zeros((3,2,3))]

    for s in range(3):
        raw_a[0][:,:,s] = np.array([
            [0.8,0.3],
            [0.2,0.7]
        ])
        raw_a[1][:,:,s] = ([
            [1.0,0.0],
            [0.0,1.0],
            [1.0,1.0]
        ])
    
    
    u = jnp.array([
        [0,0],
        [0,1],
        [1,2]
    ])
    
    vec_a,vec_b,vec_d = vectorize_weights(raw_a,raw_b,raw_d,u)
    
    
    obs_u = np.zeros((Ntrials,Ntimesteps-1,3))
    obs_u[0,...] = np.array([
        [1,0,0],
        [0,1,0]
    ])
    obs_u[1,...] = np.array([
        [1,0,0],
        [1,0,0]
    ])
    obs_u = jnp.array(obs_u)
    obs_u_filter = jnp.ones_like(obs_u[...,0])
    
    Nuf = [b_f.shape[-1] for b_f in raw_b] 
    vec_transition_per_factor = vectorize_factorwise_allowable_actions(u,Nuf)
    vec_transition_history = posterior_transition_index_factor(vec_transition_per_factor,obs_u)

    filters = [jnp.array([[1.0,1.0,1.0],[1.0,1.0,1.0]]),
               jnp.array([[1.0,0.0,0.0],[1.0,1.0,1.0]])] 
    
    # Trial 1 :
    o_d_1 = [jnp.array([
            [0.9,0.1],
            [0.8,0.2],
            [0.0,1.0]
        ]),jnp.array([
            [0.9,0.1,0.0],
            [1.0,1.0,1.0],
            [1.0,1.0,1.0]
        ])]
    
    # Trial 2 :
    o_d_2 = [jnp.array([
            [0.9,0.1],
            [0.2,0.8],
            [0.0,1.0]
        ]),jnp.array([
            [0.9,0.1,0.0],
            [1.0,0.0,0.0],
            [1.0,0.0,0.0]
        ])]
    obs = [jnp.stack([o1,o2],axis=0) for o1,o2 in zip(o_d_1,o_d_2)]
    
    # smoothd_post = smooth_trial_window(obs,obs_u,
    #               filters,
    #               vec_a,vec_b,vec_d)
    # print(np.round(np.array(smoothd_post),2))
    
    # param_updater = partial(get_parameter_update,Ns=tuple(Ns_all),Nu = u.shape[0],generalize_fadeout_function = None)
    # da,db,dc,dd,de = vmap(param_updater)(obs,vec_transition_history,smoothd_post)
    # print(da[0].shape)
    # print(da[1].shape)
    alpha = 1.0
    
    transition_generalizer =None# (lambda x : jnp.exp(-alpha*x))
    
    print(obs_u.shape)
    (final_a,final_b,final_d),final_post,final_ll = EM_jax(obs,filters,obs_u,obs_u_filter,
           raw_a,raw_b,raw_d,u,
           lr_a=1.0,lr_b=1.0,lr_d=1.0 ,
           is_learn_a = True,is_learn_b = True,is_learn_d = True,
           transition_generalizer=transition_generalizer)
    
    print(final_a)
    print(np.round(np.array(final_post),2))
    print(final_post.shape)
    print(final_ll[...,0])
    exit()

