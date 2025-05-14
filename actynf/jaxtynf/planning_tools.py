import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.tree_util import tree_map
from jax import lax,vmap, jit

from functools import partial
from itertools import product

from .jax_toolbox import spm_wnorm,_jaxlog


# EFE CALCULATIONS -------------------------------------------------------
def compute_novelty(M,multidim=False):
    HARDLIMIT = 1e-10 
            # Below this prior weight, novelty is no longer computed.
            # --> Trying to avoid huge overflows in high uncertainty situations ?
    if multidim:  # If M is a list of matrices
        wM = tree_map(lambda m : (m>HARDLIMIT)*spm_wnorm(m),M)
    else :
        wM = (M>HARDLIMIT)*spm_wnorm(M)
    return wM

def compute_info_gain(qs,A):

    def info_gain_modality(A_m):
        # Likelihood entropy :        
        H_A_m = (A_m*_jaxlog(A_m)).sum(axis=0)
                
        info_gain_m = (qs*H_A_m).sum()
        return info_gain_m
    
    info_gain_all_m = tree_map(info_gain_modality,A)
    return jnp.stack(info_gain_all_m).sum()

def compute_risk(t_pref, qo, C, timedim=False):
    # t_pref is the time at which the predicted observation will occur
    
    def risk_modality(qo_m,C_m):
        pref_m_depends_on_time = (C_m.ndim>1)
        # print(pref_m_depends_on_time)
            # if C.ndim = 1, the preferences are time invariant, t_pref has no purpose
            # if C.ndim = 2, the preferences are time dependent
            
        if pref_m_depends_on_time:
            preference_m_t = C_m[...,t_pref]
        else : 
            preference_m_t = C_m
        
        risk_m_t = (qo_m*preference_m_t).sum() - (qo_m*_jaxlog(qo_m)).sum()
                            # Utility      -        Observation entropy    
        return risk_m_t
        
    risk_all_m = tree_map(risk_modality,qo,C)
    return jnp.stack(risk_all_m).sum()

def _deprecated_compute_obs_novelty(qo,qs,Anovelty) :
    """ Old version with unwanted edge-cases ?"""
    def novelty_mod(qo_m,Anovelty_m):
        modality_joint_predictive = jnp.einsum("i,j->ij",qo_m,qs)
        return (modality_joint_predictive*Anovelty_m).sum()
    observation_novelty_all_m = tree_map(novelty_mod,qo,Anovelty)
    return jnp.stack(observation_novelty_all_m).sum()

def compute_observation_novelty(qs,A,Anovelty):
    # Instead of working on the observation distribution qo, which 
    # encodes joint observation probability for future states
    # we compute the novelty for the observations of each separate state : 
    # This avoids high uncertainty cases where the predicted state dist
    # has high entropy (e.g. [0.5,0.5]) and results in a high entropy 
    # observation dist ([0.5,0.5]),  for a quite well defined likelihood mapping :
    # A = [  0.01  0.99]
    #     [  0.99  0.01]
    # W = [ -10.0 -0.01]
    #     [ -0.01 -10.0]
    # thus marginalizing over the state 
    # dimension and providinb biaised novelty predictions.
    def _mod_scan(A_nov_m,A_m):
        return jnp.einsum("ij,j->i",A_nov_m*A_m,qs).sum()
    novelty_all_mods = tree_map(_mod_scan,Anovelty,A)
    return -jnp.stack(novelty_all_mods).sum()
    
def _deprecated_compute_transition_novelty(action_vector,qs_next,qs_prev,Bnovelty) :
    transition_novelty = -jnp.einsum("i,ijk,j,k->",qs_next,Bnovelty,qs_prev,action_vector)
    # -(qs_next*(Bnovelty @ qs_prev)).sum()
    return transition_novelty

def compute_transition_novelty(action_vector,qs_prev,B_norm,B_novelty):
    prob_novel = -jnp.einsum("ijk,j,k",B_norm*B_novelty,qs_prev,action_vector)
    return prob_novel.sum()

# @partial(jit,static_argnames=['efe_a_nov','efe_b_nov'])
def compute_Gt_array(t,qo_next,qs_next,qs_prev,action_vect,
                    A,Anovelty,
                    B,Bnovelty,
                    C,
                    efe_a_nov,efe_b_nov,old_efe_computation):
    """ 
    Agent goal : plan actions to minimize this !
    """
    # We are looking for the best action at timestep t
    # To do this, let's compute the expected risk associated with observations
    # at time t+1 given our perception of the situation : 
    t_next = t + 1 # Sequential time
    risk = compute_risk(t_next,qo_next, C)
    info_gain = compute_info_gain(qs_next,A)
    
    novelty_A = 0.0
    if efe_a_nov:
        if old_efe_computation:
            novelty_A = _deprecated_compute_obs_novelty(qo_next,qs_next,Anovelty)
        else :
            novelty_A = compute_observation_novelty(qs_next,A,Anovelty)
    novelty_B = 0.0
    if efe_b_nov:
        if old_efe_computation :
            novelty_B = _deprecated_compute_transition_novelty(action_vect,qs_prev,qs_next,Bnovelty)
        else :
            novelty_B = compute_transition_novelty(action_vect,qs_prev,B,Bnovelty)
        
    # jax.debug.print("EFE component [t={}]: {}", t,jnp.stack([risk,info_gain,novelty_A,novelty_B]))
    return jnp.stack([risk,info_gain,novelty_A,novelty_B])

def autoexpand_preference_matrix(C_base,Th,option="nopref"):
    islist=(type(C_base)==list)
    
    def expand_mod(C_base_m):
        if C_base_m.ndim>1: # Preference matrix should have a temporal dimension to be autoexpanded
            if option=="nopref":
                pref_expansion = jnp.zeros((C_base_m.shape[-2],Th))
            elif option=="last":
                last_timestep = C_base_m[:,-1]
                pref_expansion = jnp.repeat(jnp.expand_dims(last_timestep,-1),Th,-1)
            return jnp.concatenate([C_base_m,pref_expansion],axis=-1)
        return C_base_m
    
    if islist:
        return tree_map(expand_mod,C_base)
    else : 
        return expand_mod(C_base)