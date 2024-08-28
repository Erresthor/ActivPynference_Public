import random as ra
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.tree_util import tree_map
from jax import lax,vmap, jit

from functools import partial
from itertools import product

import tensorflow_probability.substrates.jax.distributions as tfd

def none_like_tree(target):
    return tree_map(lambda x: None, target)

def zero_like_tree(target):
    return tree_map(lambda x: jnp.zeros_like(x), target)

def tensorify(*args):
    """ A very ugly function that transforms numpy arrays into jax tensors, while conserving list structures."""
    all_results = []
    for arg in args :
        if type(arg)==list :
            # We return a list of tensorified args :
            return_list = []
            for el in arg :
                return_list.append(tensorify(el))
            all_results.append(return_list)
        else :
            all_results.append(jnp.array(arg))
    
    if len(all_results)==1:
        return all_results[0]
    else :
        return tuple(all_results)



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

def _condition_on(probs, ll, eps = 1e-10):
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
    new_probs, norm = _normalize(new_probs + eps)
                # We add eps here in the case the new probs are very low everywhere. This simulate a "lost" agent
                # with no idea where he is since he encountered a contradiction !
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

# To generalize action mappings in linear state spaces
def weighted_padded_roll(matrix,generalize_fadeout):
    assert matrix.ndim == 2,"Weighted Padded Roll only implemented for 2D arrays"
    K = matrix.shape[0]
    roll_limit = K
    
    padded_matrix = jnp.pad(matrix,((K,K),(K,K)),mode="constant",constant_values=0)
     
    rolling_func = lambda k : jnp.roll(padded_matrix,k,[-1,-2])*generalize_fadeout(jnp.abs(k))
    
    all_rolled = vmap(rolling_func)(jnp.arange(-roll_limit,roll_limit+1))
    
    # Remove padding : 
    all_rolled = all_rolled[...,K:-K,K:-K]
    
    new_db = all_rolled.sum(axis=-3)
    
    return new_db


# Soft ranking approximations to allow differentiating under a sorting condition : 
# Work in progress, see test_sophistacted_treesearch.py (local branch)
def soft_rank(X,temp=1.0):
    """
    Soft ranking of elements in x.
    :param X: Input tensor of shape (n,).
    :param temperature: Temperature parameter for the softmax.
    :return: Soft ranking of x.
    """
    D = (X[:, None] - X[None, :])/temp
    D = jax.nn.softmax(D,axis=-1)
    print(D)
    for k in range(10):
        D /= D.sum(axis=-1,keepdims=True)
        D /= D.sum(axis=-2,keepdims=True)
    print(D)
    exit()
    n = X.shape[0]
    Y = jnp.arange(n)
    pairwise_distances = jax.vmap(jax.vmap(lambda x,y: x-y, (0, None)), (None, 0))(X,Y)
    print(pairwise_distances)
    print(jax.nn.softmax(pairwise_distances/temp,axis=0))

def sinkhorn(a,b,x,y,dist,temp,N):
    n = x.shape[0]
    
    print(x)
    print(y)
    C = vmap(vmap(dist,(0,None)),(None,0))(x,y)
    print(C)
    K = jnp.exp(-C/temp)
    
    u = jnp.ones((n,))
    
    for t in range(N):
        v = b/jnp.dot(K.T,u)
        u = a/jnp.dot(K,v)
    print(v)
    print(u)
    eldoto = jnp.dot(jnp.dot(jnp.diag(u),K),jnp.diag(v))
    # print(*K*jnp.diag(v))
    print(eldoto)
    
if __name__=="__main__": 
    X = jnp.array([0.5,0.3,0.2,0.0,0.0])
    Y = jnp.arange(X.shape[0]).astype(float)
    temp = 0.01
    source_dist = np.array([1, 1, 1, 1, 1])
    target_dist = np.array([1, 1, 1, 1, 1])
    distance_func = lambda x,y:(x-y)*(x-y)
    N = 20
    
    sinkhorn(source_dist,target_dist,X,Y,distance_func,temp,N)
    
    exit()
    print(soft_rank(a,temp=1.0))
    exit()
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
    
    