import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax,vmap, jit
from jax.tree_util import tree_map
from functools import partial,reduce

from .jax_toolbox import _normalize,_swapaxes,_jaxlog

from .planning_tools import compute_novelty

# A collection of functions to change the form of the hidden state space (vectorized vs factorized)


# Kronecker operations, useful to get from vectorized to factorized state space !
def jax_add_kron(a, b):
    return jnp.kron(a,jnp.ones_like(b)) + jnp.kron(jnp.ones_like(a),b)

def kron_prod(list_of_tensors):
    _kron_product = 1.0
    for tensor_idx,tensor in enumerate(list_of_tensors):
        _kron_product = jnp.kron(_kron_product,tensor)
    return _kron_product

def kron_sum(list_of_tensors):
    _kron_sum = 0.0
    for tensor_idx,tensor in enumerate(list_of_tensors):
        _kron_sum = jax_add_kron(_kron_sum,tensor)
    return _kron_sum

def sum_except_axis(array_to_sum, axis):
    """
    Sums the elements of the array along all axes except the specified axis.

    Parameters:
    array (jax.numpy.ndarray): Input array.
    axis (int): The axis to keep.

    Returns:
    jax.numpy.ndarray: An array with the sum along all axes except the specified one.
    """
    # Get the total number of dimensions in the array
    num_dims = array_to_sum.ndim
    
    # Create a tuple of all axes except the specified one
    axes_to_sum = tuple(i for i in range(num_dims) if i != axis)
    
    # Sum along the specified axes
    result = jnp.sum(array_to_sum, axis=axes_to_sum)
    
    return result

# Operations on belief tensors
@jit
def to_vec_space(list_of_tensors):
    return kron_prod(list_of_tensors)

def marginalize_along_all_dims(distribution):
    axes = list(range(distribution.ndim))
    
    def marginalize_along_dim_x(x):
        
        return sum_except_axis(distribution,x)
    
    mapped_dists = tree_map(marginalize_along_dim_x,axes)
    return mapped_dists

@partial(jit,static_argnames=['source_space_shape'])
def to_source_space(tensor_1d,source_space_shape):
    tensor_multidim = jnp.reshape(tensor_1d,source_space_shape,order="C")
    return marginalize_along_all_dims(tensor_multidim)


# Operation on mappings
@jit
def to_vec_space_a(a_matrix):
    # TODO : this as a tree_map
    return [jnp.reshape(a_mod,(a_mod.shape[0],-1),order="C") for a_mod in a_matrix]

def to_source_space_a(a_matrix,latent_space_shape):
    return [jnp.reshape(a_mod,(a_mod.shape[0],)+latent_space_shape,order="C") for a_mod in a_matrix]
    
@jit
def norm_to_vec_a(a_matrix):
    """ Transform a list of a matrices to a list of normalized 2D tensors for state inference & planning."""
    a_norm = _normalize(a_matrix,tree=True)
    return [jnp.reshape(a_mod,(a_mod.shape[0],-1),order="C") for a_mod in a_norm]

@jit
def norm_to_vec_b(b_matrix,u):
    """ Transform a list of b matrices and a set of allowable actions to a 3D tensor 
    encoding possible flattened state transitions. 
    This application is not reversible. 
    """
    Nf = len(b_matrix)
    b_norm = _normalize(b_matrix,tree=True)    
        
    # "Flatten" in a single latent dimension using a kronecker form
    # For the novelty, we add the novelties together
    # Kronecker sum of the novelties : 
    flat_b = []
    for u_idx,act in enumerate(u):
        action_factor_transition_list = [b_norm[f_idx][...,act[f_idx]] for f_idx in range(Nf)]         
        flat_b.append(kron_prod(action_factor_transition_list))
    return jnp.stack(flat_b,axis=-1)

@jit
def to_log_space(c,e):
    # Preference matrix : 
    def logspace_c_mod(c_m):
        return _jaxlog(jax.nn.softmax(c_m,axis=0))
    log_c = tree_map(logspace_c_mod,c)
    log_e = _jaxlog(_normalize(e)[0])
    return log_c,log_e

@partial(jit,static_argnames=["compute_a_novelty",'compute_b_novelty'])
def get_vectorized_novelty(raw_a,raw_b,u,
                      compute_a_novelty=False,
                      compute_b_novelty=False):
    Nf = len(raw_b)
    
    if u.ndim==1:
        u = jnp.expand_dims(u,-1)
    
    a_novelties = None
    if compute_a_novelty:
        a_vec_space = to_vec_space_a(raw_a)   
        a_novelties = compute_novelty(a_vec_space,True)        

    b_novelties = None
    if compute_b_novelty:  
        b_novelties = compute_novelty(raw_b,True) # Still in list form ;)
        flat_w_b = []
        for u_idx,act in enumerate(u):
            action_factor_novelty_list = [b_novelties[f_idx][...,act[f_idx]] for f_idx in range(Nf)]
            flat_w_b.append(kron_sum(action_factor_novelty_list))
        b_novelties = (jnp.stack(flat_w_b,axis=-1))        
        
    return a_novelties,b_novelties  
        
@jit
def vectorize_weights(raw_a,raw_b,raw_d,u):
    """
    Modify the matrices so that only one latent dimension remains, 
    without changing the dynamics inherent to multiple factors
    This method normalizes the respective inputs !
    
    TODO : there should be a battery of shape testing functions here ! 
    """
    Nf = len(raw_b)
    # c is unchanged
    # e is unchanged
    
    # a : we flatten the latent state dimensions : (basically just a reshaping)
    a_vec_space = to_vec_space_a(raw_a)   
    a_norm_vec = _normalize(a_vec_space,tree=True)
    
    # b & d :  this is more complex and involves a kronecker product, which is
    # a way of "compressing" multiple factors into a single matrix 
    # Slightly different from Matlab script, because our kronecker product orders dimension differently
    assert type(raw_b)==list,"b should be a list in order to vectorize"
    b_norm = _normalize(raw_b,tree=True)    
        
    # "Flatten" in a single latent dimension using a kronecker form
    # For the novelty, we add the novelties together
    if u.ndim==1:
        u = jnp.expand_dims(u,-1)
        
    # Kronecker product 
    flat_b = []
    for u_idx,act in enumerate(u):
        action_factor_transition_list = [b_norm[f_idx][...,act[f_idx]] for f_idx in range(Nf)]         
        flat_b.append(kron_prod(action_factor_transition_list))
    b_norm_vec = jnp.stack(flat_b,axis=-1)
    
    
    
    
    # d : Kronecker product of the initial state : 
    d_norm = _normalize(raw_d,tree=True)
    
    d_kron = 1.0
    for f_idx,d_f in enumerate(d_norm):
        d_kron = jnp.kron(d_kron,d_f)
    d_norm_vec = d_kron
    # return jnp.sum(b_novelties)
    
    return a_norm_vec,b_norm_vec,d_norm_vec

