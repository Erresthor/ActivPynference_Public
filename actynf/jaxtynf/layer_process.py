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

from .jax_toolbox import _normalize,_jaxlog,random_split_like_tree,none_like_tree

# STATES __________________________________________________________________________________________
def sample_initial_state(rngkey,D):
    
    initial_state_idx = tfd.Categorical(probs=D).sample(seed=rngkey)
    
    return D,initial_state_idx,jax.nn.one_hot(initial_state_idx,D.shape[-1])

def sample_next_state(rngkey,s_previous,B,u_vect):    
    Ns = s_previous.shape[-1]
    
    new_s_d = jnp.einsum("iju,j,u->i",B,s_previous,u_vect)
    
    new_s_idx = tfd.Categorical(probs = new_s_d).sample(seed=rngkey)
    
    new_s_vect = jax.nn.one_hot(new_s_idx,Ns)
    
    return new_s_d,new_s_idx,new_s_vect



# EMISSIONS ______________________________________________________________________________________
def sample_observation(rngkey,A,s) :
    """ Old version """
    Nos = tree_map(lambda x : x.shape[-2],A)
    
    def compute_o_d_mod(A_m):
        o_mod_d = A_m@s
        return o_mod_d
    
    def sample_o_mod(o_mod_d,rng_key_m):
        o_mod_idx = tfd.Categorical(probs = o_mod_d).sample(seed=rng_key_m)
        return o_mod_idx
    
    def vectorize_o_mod(o_idx_m,No_m):
        return jax.nn.one_hot(o_idx_m,No_m)
    
    tree_like_keys = random_split_like_tree(rngkey,A)
    new_o_d = tree_map(compute_o_d_mod,A)
    new_o_idx = tree_map(sample_o_mod,new_o_d,tree_like_keys)
    new_o_vect = tree_map(vectorize_o_mod,new_o_idx,Nos)
    return new_o_d,new_o_idx,new_o_vect

# STATES + EMISSIONS _____________________________________________________________________________
def initial_state_and_obs(rngkey,A,D):
    rngkey,d_key,a_key = jr.split(rngkey,3)

    s_d,s_idx,s_vect = sample_initial_state(d_key,D)
    o_d,o_idx,o_vect = sample_observation(a_key,A,s_vect)
    return [s_d,s_idx,s_vect],[o_d,o_idx,o_vect]

def process_update(rngkey,s_previous,A,B,u_vect):
    rngkey,s_key,o_key = jr.split(rngkey,3)
    
    new_s_d,new_s_idx,new_s_vect = sample_next_state(s_key,s_previous,B,u_vect)
    
    new_o_d,new_o_idx,new_o_vect = sample_observation(o_key,A,new_s_vect)
    
    return [new_s_d,new_s_idx,new_s_vect],[new_o_d,new_o_idx,new_o_vect]


# MANAGING USER DEFINED OUTCOMES ________________________________________________________________
def sample_state(key,t,previous_s_vect,previous_u_vect,
                 B,D):
    """ 
    Sample the next state in the HMM. If this is the first timestep, sample from the D matrix. 
    Static parameter t (obviously)
    """
    key, state_key = jr.split(key)
    if t==0:
        new_s_d,new_s_idx,new_s_vect = sample_initial_state(state_key,D)
    else :
        new_s_d,new_s_idx,new_s_vect = sample_next_state(state_key,previous_s_vect,B,previous_u_vect)
    return new_s_d,new_s_idx,new_s_vect

def sample_emission_modality(rngkey,A_m,s):
    No_m = A_m.shape[-2]
    
    o_mod_d = A_m@s
    o_mod_idx = tfd.Categorical(probs = o_mod_d).sample(seed=rngkey)
    o_mod_vect = jax.nn.one_hot(o_mod_idx,No_m)
    return o_mod_d,o_mod_idx,o_mod_vect

def sample_emission(rngkey,A,s):
    tree_like_keys = random_split_like_tree(rngkey,A)
    
    reduced_function = (lambda k,a : sample_emission_modality(k,a,s))
    mapped_over_modalities = tree_map(reduced_function,tree_like_keys,A)
    
    # Transpose the list of tuples to get a tuple of lists
    mapped_o_d,mapped_o_idx,mapped_o_vect = map(list,zip(*mapped_over_modalities))
    
    return mapped_o_d,mapped_o_idx,mapped_o_vect
    # return mapped_o_d,mapped_o_idx,mapped_o_vect

def check_fixed_outcome(potential_tensor,t):
    """ 
    Does not work with scan ! 
    I don't know how to set observations / emissions 
    
    """
    try :
        assert potential_tensor[t] >= 0, 'This is an acceptable input'
        return True
    except :
        return False

def fetch_outcome(rngkey,
            t,previous_s_vect,previous_u_vect,
            A,B,D,
            fixed_states_array=None,fixed_outcomes_tree=None):   
    """ 
    Checks if a state or an outcome was predefined. If not, generate one following the MDP structure.
    This method cannot be used in vectorized functions (such as scan or vmap).
    """
    Ns = D.shape[-1]
     
    # 1. Are the states defined by the user ?
    # Note : does not work with jax scans :(
    if check_fixed_outcome(fixed_states_array,t):
        new_s_idx = fixed_states_array[t]
        new_s_d = jax.nn.one_hot(new_s_idx,Ns)
        new_s_vect = jax.nn.one_hot(new_s_idx,Ns)
    else : # Else, generate them following the usual HMM formulation
        rngkey,state_key = jr.split(rngkey)
        new_s_d,new_s_idx,new_s_vect = sample_state(state_key,t,previous_s_vect,previous_u_vect,B,D)
       
    # 2. Are the emissions defined by the user ?
    if fixed_outcomes_tree==None:
        fixed_outcomes_tree = none_like_tree(A)
    rngkeys_all_mods = random_split_like_tree(rngkey,A)
    
    def generate_obs_mod(_rng_key_m,A_m,_fixed_outc_m):
        _No_m = A_m.shape[-2]
        if check_fixed_outcome(_fixed_outc_m,t):
            new_o_idx_mod = _fixed_outc_m[t]
            return jax.nn.one_hot(o_mod_idx,_No_m),new_o_idx_mod,jax.nn.one_hot(o_mod_idx,_No_m)
        else : # Else, generate them following the usual HMM formulation
            o_mod_d,o_mod_idx,o_mod_vect = sample_emission_modality(_rng_key_m,A_m,new_s_vect,_No_m)      
            return o_mod_d,o_mod_idx,o_mod_vect
        
    mapped_over_modalities = tree_map(generate_obs_mod,rngkeys_all_mods,A,fixed_outcomes_tree)
    
    # Transpose the list of tuples to get a tuple of lists
    new_o_d,new_o_idx,new_o_vect = map(list,zip(*mapped_over_modalities)) 
    
    return [new_s_d,new_s_idx,new_s_vect],[new_o_d,new_o_idx,new_o_vect]


# BASED ON PYRO
# Useful for inference when no action can be seen ?
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