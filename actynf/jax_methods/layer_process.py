import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import jax
from jax.tree_util import tree_map
from jax import lax
from jax import vmap
from jax import jit

from numpyro import plate,sample,deterministic
import numpyro.distributions as distr

import tensorflow_probability.substrates.jax.distributions as tfd

from .jax_toolbox import _normalize,_jaxlog,random_split_like_tree




def sample_initial_state(rngkey,D,Ns):
    initial_state_idx = tfd.Categorical(probs=D).sample(seed=rngkey)
    
    return D,initial_state_idx,jax.nn.one_hot(initial_state_idx,Ns)

def sample_next_state(rngkey,s_previous,B,u_vect,Ns):    
    new_s_d = jnp.einsum("iju,j,u->i",B,s_previous,u_vect)
    
    new_s_idx = tfd.Categorical(probs = new_s_d).sample(seed=rngkey)
    
    new_s_vect = jax.nn.one_hot(new_s_idx,Ns)
    
    return new_s_d,new_s_idx,new_s_vect

def sample_observation(rngkey,A,s,Nos) :
    def compute_o_d_mod(A_m):
        o_mod_d = A_m@s
        return o_mod_d
    
    def sample_o_mod(o_mod_d,rng_key_m):
        o_mod_idx = tfd.Categorical(probs = o_mod_d).sample(seed=rng_key_m)
        return o_mod_idx
    
    def vectorize_o_mod(A_m,o_idx_m,No_m):
        return jax.nn.one_hot(o_idx_m,No_m)
    
    tree_like_keys = random_split_like_tree(rngkey,A)
    new_o_d = tree_map(compute_o_d_mod,A)
    new_o_idx = tree_map(sample_o_mod,new_o_d,tree_like_keys)
    new_o_vect = tree_map(vectorize_o_mod,A,new_o_idx,Nos)
    return new_o_d,new_o_idx,new_o_vect

def initial_state_and_obs(rngkey,A,D,Ns,Nos):
    rngkey,d_key,a_key = jr.split(rngkey,3)

    s_d,s_idx,s_vect = sample_initial_state(d_key,D,Ns)
    o_d,o_idx,o_vect = sample_observation(a_key,A,s_vect,Nos)
    return [s_d,s_idx,s_vect],[o_d,o_idx,o_vect]

def process_update(rngkey,s_previous,A,B,u_vect,Ns,Nos):
    rngkey,s_key,o_key = jr.split(rngkey,3)
    
    new_s_d,new_s_idx,new_s_vect = sample_next_state(s_key,s_previous,B,u_vect,Ns)
    
    new_o_d,new_o_idx,new_o_vect = sample_observation(o_key,A,new_s_vect,Nos)
    
    return [new_s_d,new_s_idx,new_s_vect],[new_o_d,new_o_idx,new_o_vect]

def process_update_pyro(s_previous,A,B,u_vect):
    """ 
    TODO : Rewrite this to incorporate with the functions above !
    """
    new_s_d = jnp.einsum("iju,j,u->i",B,s_previous,u_vect)
    
    new_s_idx = sample("true_states",distr.Categorical(probs=new_s_d))
    
    new_s_vect = jax.nn.one_hot(new_s_idx,new_s_d.shape[0])
    
    def compute_o_d_mod(A_m):
        o_mod_d = A_m@new_s_vect
        return o_mod_d
    
    def sample_o_mod(o_mod_d):
        o_mod_idx = sample("observations",distr.Categorical(probs=o_mod_d))
        return o_mod_idx
    
    def vectorize_o_mod(A_m,o_idx_m):
        No_m = A_m.shape[0]
        return jax.nn.one_hot(o_idx_m,No_m)
    
    new_o_d = tree_map(compute_o_d_mod,A)
    new_o_idx = tree_map(sample_o_mod,new_o_d)
    new_o_vect = tree_map(vectorize_o_mod,A,new_o_idx)

    return [new_s_d,new_s_idx,new_s_vect],[new_o_d,new_o_idx,new_o_vect]
    
if __name__ == '__main__':
    import random as ra
    import numpy as np
    Nos = np.array([10,3,2])
    Ns = 10
    T = 10
    Np = 10

    key = jr.PRNGKey(5002)

    fixed_observations = [np.random.randint(0,No,(T,)) for No in Nos]
    obs_vectors = [jax.nn.one_hot(rvs,No,axis=0) for rvs,No in zip(fixed_observations,Nos)]




    # A = [_normalize(jr.uniform(key,(No,Ns)))[0] for No in Nos]

    Nmod = 5
    A = [_normalize(jnp.eye(Ns))[0] for i in range(Nmod)]
    A[4] = _normalize(jr.uniform(key,(3,Ns)))[0]

    C = [jnp.zeros((a.shape[0],)) for a in A]
    C[1] = jnp.linspace(0,10,C[1].shape[0])

    # obs_vectors = [_normalize(jr.uniform(key,(No,)))[0] for No in Nos]

    B = np.zeros((Ns,Ns,Np))
    for u in range(Ns):
        B[:,:,u] = np.eye(Ns)
        try :
            B[u+1,u,u] = 0.5
            B[u,u,u] = 0.5
        except:
            continue
    B = jnp.asarray(B)    
    
    key,key_d  = jr.split(key)
    D,_ = _normalize(jr.uniform(key_d,(Ns,)))

    key,key_sample = jr.split(key)
    _,s_0 = initial_state(key_sample,D)
    
    key,key_sample = jr.split(key)
    u_vect=  jax.nn.one_hot(3,Np)
    print(u_vect)
    s,o = process_update(key_sample,s_0,A,B,jax.nn.one_hot(3,Np))
    print(o)
    # A_novel = compute_novelty(A,True)
    # B_novel = compute_novelty(B)
    # qsm,_ = _normalize(jr.uniform(key,(Ns,)))

    # key,key2 = jr.split(key)
    # qsp,_ = _normalize(jr.uniform(key2,(Ns,)))    

    # E = jnp.ones((Np,))


    # key = jr.PRNGKey(5052)
    # key,key_d  = jr.split(key)
    # D,_ = _normalize(jr.uniform(key_d,(Ns,)))

    # key,key_sample = jr.split(key)
    # s_0 = initial_state(key_sample,D)
    # # print(transition_distribution())
    # # print(s_0)

    # key,key_sample = jr.split(jr.PRNGKey(5))
    # s_1,d = sample_new_state(key_sample,s_0,B,2)
    # # print(s_1)
    # # print(d)
    

    # sample_observation(key_sample,A,s_1)
    
    
    # # os,s = process_update(key_sample,s_0,A,B,3)
    # # print(os)
    # # print(s)