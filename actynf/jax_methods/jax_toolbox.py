import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.tree_util import tree_map
from jax import lax,vmap, jit

from functools import partial
from itertools import product

import tensorflow_probability.substrates.jax.distributions as tfd

def _swapaxes(arr,ax1=-1,ax2=-2,tree=False):
    if tree : 
        return tree_map(lambda x: jnp.swapaxes(x,ax1,ax2),arr)
    return jnp.swapaxes(arr,ax1,ax2)

def _normalize_single_tensor(u, axis=0, eps=1e-15):
    u = jnp.where(u == 0, 0, jnp.where(u < eps, eps, u))
    c = u.sum(axis=axis)
    c = jnp.where(c == 0, 1, c)
    return u / c
    
# From the dynamax github :
def _normalize(u, axis=0, eps=1e-15,tree=False):
    """Normalizes the values within the axis in a way that they sum up to 1.

    Args:
        u: Input array to normalize.
        axis: Axis over which to normalize.
        eps: Minimum value threshold for numerical stability.

    Returns:
        Tuple of the normalized values, and the normalizing denominator.
    """    
    if tree :
        _normalize_mod = (lambda u_mod : _normalize(u_mod,axis,eps)[0])
        return tree_map(_normalize_mod,u)
    u = jnp.where(u == 0, 0, jnp.where(u < eps, eps, u))
    c = u.sum(axis=axis,keepdims=True)
    c = jnp.where(c == 0, 1, c)
    return u / c, c

def _condition_on(probs, ll):
    """Condition on new emissions, given in the form of log likelihoods
    for each discrete state, while avoiding numerical underflow.

    Args:
        probs(k): prior for state k
        ll(k): log likelihood for state k

    Returns:
        probs(k): posterior for state k
    """
    ll_max = ll.max()
    new_probs = probs * jnp.exp(ll - ll_max)
    new_probs, norm = _normalize(new_probs)
    log_norm = jnp.log(norm) + ll_max
    return new_probs, log_norm

# Log likelihoods like dynamax (generic for tfd distributions) :
def A_mat_to_tfd(matrix,state,inputs=None):
    return tfd.Categorical(probs=matrix[:,state])

def B_mat_to_tfd(matrix,state,inputs=None):
    action_idx = 0 # Action defaults to 0
    if inputs is not None:
        action_idx = inputs
    return tfd.Categorical(probs=matrix[:,state,action_idx])

def _compute_A_conditional_logliks(A, observations, inputs=None):
    # Compute the log probability for each time step by
    # performing a nested vmap over emission time steps and states.
    Ns = A.shape[-1]
    f = lambda o, inpt: \
        vmap(lambda state: A_mat_to_tfd(A, state, inpt).log_prob(o))(jnp.arange(Ns))
    return vmap(f)(observations, inputs)

def _compute_policy_logliks(qpi, observed_u, alpha=16.0):
    # Compute the log probability for each time step by
    # performing a nested vmap over emission time steps and states.
    f = lambda u : _jaxlog(jax.nn.softmax(alpha*qpi))[u]
    return vmap(f)(observed_u)

# Very basic functions
def _jaxlog(x,eps = 1e-10):
    """
    A slighlty modified lax.log implementation to avoid numerical overflow around 0.
    """
    return lax.log(jnp.clip(x,a_min=eps))

def spm_wnorm(A,eps=1e-10):
    """ 
    Returns Expectation of logarithm of Dirichlet parameters over a set of 
    Categorical distributions, stored in the columns of A.
    """
    A = jnp.clip(A, a_min=eps)
    norm = 1. / A.sum(axis=0)
    avg = 1. / A
    wA = norm - avg
    return wA


# EFE CALCULATIONS -------------------------------------------------------
def compute_novelty(M,multidim=False):
    HARDLIMIT = 1e-15  
            # Below this prior weight, novelty is no longer computed.
            # --> Trying to avoid huge overflows in high uncertainty situations
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

def compute_risk(qo, C, timedim=False):
    def risk_modality(qo_m,C_m):
        one_timestep_risk = lambda qo_m_t : (qo_m_t*C_m).sum() - (qo_m_t*_jaxlog(qo_m_t)).sum()
                                                # Utility      -        Observation entropy        
        if timedim:
            T =  qo_m.shape[-1]
            timefunc = lambda t: one_timestep_risk(qo_m[:,t])
            return vmap(timefunc)(jnp.arange(0,T,1))
        else : 
            return one_timestep_risk(qo_m)
    
    risk_all_m = tree_map(risk_modality,qo,C)
    return jnp.stack(risk_all_m).sum()

def compute_observation_novelty(qs,A,Anovelty):
    # Instead of working on the observation distribution qo, which 
    # encodes joint observation probability for future states
    # we compute the novelty for the observations of each separate state : 
    # This avoids high uncertainty cases where the predicted state dist
    # has high entropy (e.g. [0.5,0.5]) and results in a high entropy 
    # observation dist ([0.5,0.5]), thus marginalizing over the state 
    # dimension and providinb biaised novelty predictions.
    def _mod_scan(A_nov_m,A_m):
        return jnp.einsum("ij,j->i",A_nov_m*A_m,qs).sum()
    novelty_all_mods = tree_map(_mod_scan,Anovelty,A)
    return -jnp.stack(novelty_all_mods).sum()


def _deprecated_compute_obs_novelty(qo,qs,Anovelty) :
    """ Old version with unwanted edge-cases"""
    def novelty_mod(qo_m,Anovelty_m):
        modality_joint_predictive = jnp.einsum("i,j->ij",qo_m,qs)
        return (modality_joint_predictive*Anovelty_m).sum()
    observation_novelty_all_m = tree_map(novelty_mod,qo,Anovelty)
    return jnp.stack(observation_novelty_all_m).sum()
    
def _deprecated_compute_transition_novelty(action_vector,qs_next,qs_prev,Bnovelty) :
    transition_novelty = -jnp.einsum("i,ijk,j,k->",qs_next,Bnovelty,qs_prev,action_vector)
    # -(qs_next*(Bnovelty @ qs_prev)).sum()
    return transition_novelty

def compute_transition_novelty(action_vector,qs_prev,B_norm,B_novelty):
    prob_novel = -jnp.einsum("ijk,j,k",B_norm*B_novelty,qs_prev,action_vector)
    return prob_novel.sum()

def compute_Gt_array(qo_next,qs_next,qs_prev,action_vect,
                    A,Anovelty,
                    B,Bnovelty,
                    C):
    """ 
    Agent goal : plan actions to minimize this !
    """
    risk = compute_risk(qo_next, C)
    info_gain = compute_info_gain(qs_next,A)
    
    novelty_A = compute_observation_novelty(qs_next,A,Anovelty)
    novelty_B = compute_transition_novelty(action_vect,qs_prev,B,Bnovelty)
    return jnp.stack([risk,info_gain,novelty_A,novelty_B])

def random_split_like_tree(rng_key, target=None, treedef=None):
    """
    From : https://github.com/google/jax/discussions/9508
    """
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)

def convert_to_one_hot_list(list_of_idxs,list_of_shapes):
    mapped_func = (lambda x_idx,x_shape : jax.nn.one_hot(x_idx,x_shape))
    return tree_map(mapped_func,list_of_idxs,list_of_shapes)

if __name__=="__main__":
    import random as ra
    import numpy as np
    Nos = np.array([10,3,2])
    Ns = 10
    T = 10
    Np = 10

    key = jr.PRNGKey(550000)

    fixed_observations = [np.random.randint(0,No,(T,)) for No in Nos]
    obs_vectors = [jax.nn.one_hot(rvs,No,axis=0) for rvs,No in zip(fixed_observations,Nos)]

    # A = [_normalize(jr.uniform(key,(No,Ns)))[0] for No in Nos]

    Nmod = 2
    A = [_normalize(jnp.eye(Ns))[0] for i in range(Nmod)]
    

    C = [jnp.zeros((a.shape[0],)) for a in A]
    C[1] = jnp.linspace(0,10,C[1].shape[0])

    obs_vectors = [_normalize(jr.uniform(key,(No,)))[0] for No in Nos]

    B = np.zeros((Ns,Ns,Np))
    for u in range(Ns):
        B[:,:,u] = np.eye(Ns)
        try :
            B[u+1,u,u] = 0.5
            B[u,u,u] = 0.5
        except:
            continue

    qsm,_ = _normalize(jr.uniform(key,(Ns,)))
    qsp,_ = _normalize(jr.uniform(key,(Ns,)))    


    
    
    # Gt = compute_Gt_array(obs_vectors,qsp,qsm,
    #                  A,compute_novelty(A,True),
    #                  compute_novelty(B,False),C)

    # print(compute_G_pi(qpi,qsm,A,B))
    
    A_novel = compute_novelty(A,True)
    B_novel = compute_novelty(B)
    # A_novel = [jnp.zeros(a.shape) for a in A_novel]
    # B_novel = jnp.zeros(B_novel.shape)
    E = jnp.ones((Np,))




    # Planning next action for the next 6 tmstps !
    # key, subkey = jr.split(key)  # Create a random seed for SKLearn.  

    key = jr.PRNGKey(ra.randint(0,100000))
    Th = 15    # qpi_allts,_ = _normalize(jr.uniform(key,(Np,Th)))
    qpi_allts,_ = _normalize(jnp.ones((Np,Th)))
    # qsm,_ = _normalize(jr.uniform(key,(Ns,)))
    qsm = jax.nn.one_hot(3,Ns)
    # qsm,_ = _normalize(jnp.ones((Ns,)))
    # print(qsm)
    # print(qpi_allts)
    
    