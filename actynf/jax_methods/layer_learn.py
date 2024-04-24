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

from jax_toolbox import _normalize,_jaxlog
from jax_toolbox import compute_Gt_array,compute_novelty
from jax_toolbox import _condition_on,_compute_A_conditional_logliks

def learn_a(hist_obs,hist_qs,pa,lr_a):
    def _learn_a_mod(o_mod,pa_mod):
        da = jnp.einsum("it,jt->ijt",o_mod,hist_qs)
        return pa_mod + lr_a*da.sum(axis=-1)
    return tree_map(_learn_a_mod,hist_obs,pa)

def learn_b(hist_u,hist_qs,pb,lr_b):
    post_qs = hist_qs[:,1:]
    pre_qs = hist_qs[:,:-1]
    db = jnp.einsum("it,jt,ut->ijut",post_qs,pre_qs,hist_u)
    return pb + lr_b*db.sum(axis=-1)

def learn_d(hist_qs,pd,lr_d):
    return pd + lr_d*hist_qs[:,0]

def smooth_posterior_after_trial(hist_qs):
    """
    Not yet implemented !!
    TODO : Implement a Forwards-Backwards smoother here !
    For now, the smoothing does not change the state posteriors !
    """
    return hist_qs

def learn_after_trial(hist_obs_vect,hist_qs,hist_u_vect,
          pa,pb,pd,
          learn_what={"a":True,"b":True,"d":True},
          learn_rates={"a":1.0,"b":1.0,"d":1.0},
          post_trial_smooth = True):
    # T is the last dimension, check that input vectors have the right shape !
    
    hist_qs_loc = hist_qs
    if post_trial_smooth:
        hist_qs_loc  = smooth_posterior_after_trial(hist_qs)
    
    a = pa
    if learn_what["a"]:
        a = learn_a(hist_obs_vect,hist_qs_loc,pa,learn_rates["a"])
    else : 
        a = pa
        
    
    if learn_what["b"]:
        b = learn_b(hist_u_vect,hist_qs_loc,pb,learn_rates["b"])
    else : 
        b = pb
        
    if learn_what["d"]:
        d = learn_d(hist_qs_loc,pd,learn_rates["d"])
    else : 
        d = pd
    return a,b,d

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

