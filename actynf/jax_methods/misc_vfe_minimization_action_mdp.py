# -*- coding: utf-8 -*-
"""
A Jax implementation of the forwards-backwards algorithm from dynamax HMM module
but with multiple possible transitions per timestep depending on agent decisions.
________________________________________________________________

Created on 12/02/24

@author: Côme ANNICCHIARICO(come.annicchiarico@inserm.fr)

re-implementing dynamax hmm/inference.py functions
"""

import numpy as np
import time
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import tree_map
from jax.scipy.special import gammaln,digamma,betaln
from tensorflow_probability.substrates import jax as tfp
from functools import partial

import optax

import matplotlib.pyplot as plt
import functools 

from .jax_toolbox import _jaxlog,_normalize

# Function from pymdp :
def spm_wnorm(A,epsilon=1e-10):
    """ 
    Returns Expectation of logarithm of Dirichlet parameters over a set of 
    Categorical distributions, stored in the columns of A.
    """
    A = jnp.clip(A, a_min=epsilon)
    norm = 1. / A.sum(axis=0)
    avg = 1. / A
    wA = norm - avg
    return wA

def dirichlet_expected_value(dir_arr,epsilon=1e-10):
    """ 
    Returns Expectation of Dirichlet parameters over a set of 
    Categorical distributions, stored in the columns of A.
    """
    dir_arr = jnp.clip(dir_arr, a_min=epsilon)
    expected_val = jnp.divide(dir_arr, dir_arr.sum(axis=0, keepdims=True))
    return expected_val


# Manual training gradient descent functions
def update_params(param_idx,_X,_gradX,_lr):

    def _recursive_update(__x,__gradx):
        """ 
        Parameter update given gradient, taking account of nested 
        parameter list.
        """
        if type(__x)==list:
            assert type(__gradx)==list,"Type mismatch in update_params"
            for k,(x_elt,grad_elt) in enumerate(zip(__x,__gradx)):
                x_elt = _recursive_update(x_elt,grad_elt)
                __x[k] = x_elt
        else : 
            __x = __x - _lr*__gradx
        return __x

    X_infered = [_X[i] for i in param_idx]
    assert len(X_infered) == len(_gradX), "length mismatch between vfe variable and gradient"

    for k,(x,x_id,grx) in enumerate(zip(X_infered,param_idx,_gradX)):
        x = _recursive_update(x,grx)
        _X[x_id] = x
    return _X

# @partial(jnp.vectorize, signature='(k),(k)->()') 
# Not sure about this so we'll use vmap instead, restraining the 
# kl dirichlet computation to 2 dimsensions for now 
@partial(vmap, in_axes=(1,1,None))
def _kl_dir_dirichlet(dir1,dir2,epsilon=1e-5):
    # Analytical expression of the KL divergence between two dirichlet 
    # parameter tensors. (see equations @ https://statproofbook.github.io/P/dir-kl.html)
    # Also inspired from 
    # https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/dirichlet.py#L379
    # and https://github.com/pytorch/pytorch/blob/main/torch/distributions/kl.py#L262
    # KL[Dir(x; a) || Dir(x; b)]
    #     = E_Dir(x; a){log(Dir(x; a) / Dir(x; b)}
    #     = E_Dir(x; a){sum_i[(a[i] - b[i]) log(x[i])} - (lbeta(a) - lbeta(b))
    #     = sum_i[(a[i] - b[i]) * E_Dir(x; a){log(x[i])}] - lbeta(a) + lbeta(b)
    #     = sum_i[(a[i] - b[i]) * (digamma(a[i]) - digamma(sum_j a[j]))]
    #          - lbeta(a) + lbeta(b))
    dir1 = jnp.clip(dir1, a_min=epsilon)
    dir2 = jnp.clip(dir2, a_min=epsilon)


    sum_dir1 = dir1.sum()
    sum_dir2 = dir2.sum()

    t1 = gammaln(sum_dir1) - gammaln(sum_dir2)

    t2 = (gammaln(dir2) - gammaln(dir1)).sum()

    t3 = dir1-dir2

    t4 = digamma(dir1) - digamma(sum_dir1)
    # return (t3*t4).sum() - betaln(dir1) + betaln(dir2)
    return t1 + t2 + (t3*t4).sum()

def _kl_dir_dirichlet_list(list_1,list_2,epsilon=1e-5):
    """
        Computes the KL_dir between two lists of tensors of dirichlet parameters along axis 0.
        In : 
        - list_1 and list_2 are both lists of same length, which elements are arrays of same shape of size No x Np_1 x Np_2 x ...
        Returns : 
        - A list of arrays of size Np_1 x Np_2 x ... where each element contains the computed KL_dir for this set of parameters
    """
    def _kl_dir_one_mod(list_1_mod,list_2_mod):
        return _kl_dir_dirichlet(list_1_mod,list_2_mod,epsilon)
    kl_dir_all_mods = tree_map(_kl_dir_one_mod,list_1,list_2)
    return jnp.stack(kl_dir_all_mods)

def _logexpect_dirichlet(dir_1,epsilon=1e-5):
    dir_1 = jnp.clip(dir_1, a_min=epsilon)
    return digamma(dir_1) - digamma(dir_1.sum(axis=0))

def _logexpect_dirichlet_list(list_a,epsilon=1e-5):
    def _logexpect_dirichlet_one_mod(list_a_mod):
        return _logexpect_dirichlet(list_a_mod,epsilon)
    logexpect_all_mods = tree_map(_logexpect_dirichlet_one_mod,list_a)
    return logexpect_all_mods
    # return jnp.stack(logexpect_all_mods)

def accuracy_term(o,qs,logA):
    def _accuracy_one_mod(o_m,logA_m):
        return jnp.einsum("it,ji,jt->t",qs,logA_m,o_m)
    accuracy_all_mods = tree_map(_accuracy_one_mod,o,logA)
    return jnp.stack(accuracy_all_mods).sum(axis=0)

def act_hmm_free_energy(o, qs ,qu, A, B, D,
                               pu,pA,pB,pD,
                                epsilon_kl=1e-5):
    # the free energy for a simple hmm model with flexible transitions !

    # 1. NETWORK PARAMETER DIVERGENCES !
    # Parameter KL divergences with their respective priors
    kl_a = _kl_dir_dirichlet_list(A,pA,epsilon_kl)
    kl_b = vmap(_kl_dir_dirichlet,in_axes=(2,2,None))(B,pB,epsilon_kl)
    kl_d = _kl_dir_dirichlet(jnp.expand_dims(D,1),jnp.expand_dims(pD,1),epsilon_kl)

    # 2. HIDDEN VARIABLES (STATE + ACTION) POSTERIOR ENTROPIES
    neg_Hs = (qs*_jaxlog(qs)).sum(axis=0)   # State entropies over all timesteps / summed over all states
    neg_Hu = (qu*_jaxlog(qu)).sum(axis=0)  # Action entropies over all timesteps / summed over all allowables

    # Term I : parameter prior divergences + posterior entropies
    term1 = kl_a.sum() + kl_b.sum() + kl_d.sum() + neg_Hs.sum() + neg_Hu.sum()

    # 3. PRIOR CROSS ENTROPIES 
    # (actions only, as state priors are parametrized by B and D)
    # Negative cross entropy between qu and prior u:
    negHpu = (qu*_jaxlog(pu)).sum(axis=0)

    # 4. LIKELIHOODS !
    # Compute the log expextations of the parameters under their 
    # respective Dirichlet distribution once only :
    logA = _logexpect_dirichlet_list(A) # A list of logA for each observation modality
    logB = _logexpect_dirichlet(B) # A single logB tensor
    logD = _logexpect_dirichlet(D) # A single logD tensor

    qs_init = qs[:,0]
    log_initial = qs_init.dot(logD)

    qs_tminus = qs[:,:-1]
    qs_tplus = qs[:,1:]
    log_transition = jnp.einsum("it,ijk,jt,kt->t",qs_tplus,logB,qs_tminus,qu)
    
    log_perception = accuracy_term(o,qs,logA)
    
    # TERM II :
    term2 = log_initial + log_transition.sum() + log_perception.sum() + negHpu.sum()

    vfe = term1 - term2
    return vfe

def sample_HMM(key,
               trueA,trueB,trueD,truePu,T):
    key, subkey = jax.random.split(key)
    s1 = jax.nn.one_hot(jax.random.categorical(subkey, trueD,shape=(1,)),trueB.shape[0])[0,:]

    key, subkey = jax.random.split(key)
    os = []
    o1 = [jax.nn.one_hot(jax.random.categorical(subkey,_jaxlog(jnp.einsum("ij,j->i",a,s1)),shape=(1,)),a.shape[0])[0,:] for a in trueA]
    os.append(o1)


    s_list = [s1]
    u_list = []

    stminus = s1
    for t in range(1,T):
        key, subkey = jax.random.split(key)
        utminus = jax.nn.one_hot(jax.random.categorical(subkey,pu[:,t-1],shape=(1,)),pu.shape[0])[0,:]

        key, subkey = jax.random.split(key)
        next_log_probs = _jaxlog(jnp.einsum("ijk,j,k->i",trueB,stminus,utminus))
        st = jax.nn.one_hot(jax.random.categorical(subkey,next_log_probs,shape=(1,)),trueB.shape[0])[0,:]
        # print(stminus,st,jnp.einsum("ijk,j,k->i",trueB,stminus,utminus))

        key, subkey = jax.random.split(key)
        ot = [jax.nn.one_hot(jax.random.categorical(subkey,_jaxlog(jnp.einsum("ij,j->i",a,st)),shape=(1,)),a.shape[0])[0,:] for a in trueA]
        os.append(ot)

        
        s_list.append(st)
        stminus = st

        u_list.append(utminus)

    o_stacked = []
    s_stacked = jnp.stack(s_list)
    u_stacked = jnp.stack(u_list)
    for mod in range(len(os[0])):
        o_stacked.append(jnp.stack([k[mod] for k in os]).swapaxes(1,0))
    return o_stacked,s_stacked,u_stacked

def supervised_training_gd(os,qs,qu,
                A,B,D,
                pu,pA,pB,pD,
            schedule,lr = 0.01,STOPPER_LIST_LEN=20,eps=5e-2):
    vfe = act_hmm_free_energy(os,qs,qu,
                A,B,D,
                pu,pA,pB,pD)

    # Initial variational posteriors
    X_qs = jnp.ones_like(qs) 
    X_qu = jnp.ones_like(qu)
    # Initial hmm parameters
    X_a = [jnp.ones_like(a) for a in A]
    X_b = jnp.ones_like(B)
    X_d = jnp.ones_like(D)

    X = [X_qs,X_qu,X_a,X_b,X_d]

    def encode_params(_X):
        [_X_qs,_X_qu,_X_a,_X_b,_X_d] = _X
        _encoded_X = [jax.nn.softmax(_X_qs,0),jax.nn.softmax(_X_qu,0),[a*a for a in _X_a],_X_b*_X_b,_X_d*_X_d]
        return _encoded_X

    training_step_init_params = encode_params(X)
    fixed_simulation_params = (pu,pA,pB,pD)
    reduced_vfe = (lambda _s,_u,a,b,d : act_hmm_free_energy(*((os,) + ( _s,_u,a,b,d) + fixed_simulation_params)))

    k_previous = [999999]

    vfes = []
    for k, program in enumerate(schedule):
        print("Program : " + str(k))
        idx_param = tuple(program[0])
            # Which parameter to train in this step ?
        dvfe = jax.grad(reduced_vfe, argnums=idx_param)

        Niter = program[1]
        for ni, it in enumerate(range(Niter)):
            
            params = encode_params(X)
            vfe = reduced_vfe(*params)
            print(vfe)
            vfes.append(vfe)
            # reduced_params = [params[i] for i in idx_param]
            # print(dvfe(*params))
            grads = dvfe(*params)

            X = update_params(idx_param, X,grads,lr)

            if (len(k_previous)==STOPPER_LIST_LEN) and ((max(k_previous) - vfe) < eps):
                results = encode_params(X)
                return vfes,results
                return

            print(k_previous)
            k_previous.append(float(vfe))
            if len(k_previous)>STOPPER_LIST_LEN:
                k_previous.pop(0)

    results = encode_params(X)
    return vfes,results

def manual_gradient_descent():
    # General remark :
    # Priors should not be close to 0 :)
    T = 20
    Ns = 5
    No1 = 5
    No2 = 8
    Nu = 1

    a0 = np.eye(Ns).astype(float)
    a1 = np.random.randint(1,10,(No2,Ns)).astype(float)
    A = [a0,a1]
    
    pA = [jnp.ones_like(a) for a in A]
    pA[0] = pA[0] + jnp.eye(Ns)*0.5

    D = jnp.array(np.random.randint(1,10,(Ns,)).astype(float)) 
    pD = jnp.ones_like(D)

    B = jnp.array(np.random.randint(1,10,(Ns,Ns,Nu)).astype(float))

    B = _normalize(jnp.expand_dims(jnp.array(10*np.eye(Ns)+np.ones((Ns,Ns))),-1))[0]
    pB = jnp.ones_like(B) # + jnp.expand_dims(jnp.eye(Ns),-1)

    qs,_ = _normalize(jnp.array(np.random.randint(1,10,(Ns,T))))
    qu,_ = _normalize(jnp.array(np.random.randint(1,10,(Nu,T-1))))

    pu,_ = _normalize(jnp.ones_like(qu))

    key = jax.random.PRNGKey(9999)
    trueA,trueB,trueD = [_normalize(a)[0] for a in A],_normalize(B)[0],_normalize(D)[0]
    
    os,true_s,true_u = sample_HMM(key,trueA,trueB,trueD,pu,T)

    training_program =[ 
        [[0,1,2,3,4],2000]
    ]

    vfes2,results2 = supervised_training_gd(os,qs,qu,
                A,B,D,
                pu,pA,pB,pD,
            training_program,lr = 0.15)


    print(trueA[0])
    # print(_normalize(results[2][0])[0])
    print(_normalize(results2[2][0])[0])

    print(trueA[1])
    # print(_normalize(results[2][1])[0])
    print(_normalize(results2[2][1])[0])

    print(trueB)
    # print(_normalize(results[3])[0])
    print(_normalize(results2[3])[0])

    # print(np.round(np.array(results[0]),2))
    print(np.round(np.array(results2[0]),2))
    # plt.plot(np.linspace(0,len(vfes),len(vfes)),vfes)
    plt.plot(np.linspace(0,len(vfes2),len(vfes2)),vfes2)
    plt.show()
    # for it in range(Niter):
    #     print(dvfe_dus(qs,qu))




def fit(params, obs, loss_func, optimizer, num_steps = 100):
    """ I'm fast as fk boi """
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        loss_value, grads = jax.value_and_grad(loss_func)(params, obs)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    losses = []
    for i in range(num_steps):
        params, opt_state, loss_value = step(params, opt_state)
        losses.append(loss_value)
        if i % 25 == 0:
            print(f'step {i}, loss: {loss_value}')

    return params,losses

if __name__=="__main__":
    T = 200
    Ns = 5
    No1 = 5
    No2 = 10
    Nu = 10
        
    
    # Prior parameters
    a0 = np.eye(Ns).astype(float)
    a1 = np.random.randint(1,10,(No2,Ns)).astype(float)
    A = [a0,a1]
    pA = [jnp.ones_like(a) for a in A]
    pA[0] = pA[0] + jnp.eye(Ns)*0.5

    D = jnp.array(np.random.randint(1,10,(Ns,)).astype(float)) 
    pD = jnp.ones_like(D)

    B = np.zeros((Ns,Ns,Nu))
    for u in range(Nu):
        random_state = np.random.randint(0,Ns)
        B[random_state,:,u] = 1.0
    # plt.imshow(B[:,:,0])
    # plt.show()
    
    # exit()
    # B = _normalize(jnp.expand_dims(jnp.array(10*np.eye(Ns)+np.ones((Ns,Ns))),-1))[0]
    pB = jnp.ones_like(B) + B*0.5 # + jnp.expand_dims(jnp.eye(Ns),-1)

    qs,_ = _normalize(jnp.array(np.random.randint(1,10,(Ns,T))))
    qu,_ = _normalize(jnp.array(np.random.randint(1,10,(Nu,T-1))))

    pu,_ = _normalize(jnp.ones_like(qu))

    key = jax.random.PRNGKey(9999)
    trueA,trueB,trueD = [_normalize(a)[0] for a in A],_normalize(B)[0],_normalize(D)[0]
    
    os,true_s,true_u = sample_HMM(key,trueA,trueB,trueD,pu,T)

    
    # Network parameters : 
    # We assume that our priors are fixed
    fixed_simulation_params = (pu,pA,pB,pD)
    
    
    # Initial variational posteriors
    X_qs = jnp.ones_like(qs) 
    X_qu = jnp.ones_like(qu)
    # Initial hmm parameters
    X_a = [jnp.ones_like(a) for a in A]
    X_b = jnp.ones_like(B)
    X_d = jnp.ones_like(D)
    
    # The initial parameters are a set of tensors  (should we randomize them ?)
    X = [X_qs,X_qu,X_a,X_b,X_d]

    # That we encode to make them acceptable parameters for our HMM model
    def encode_params(_X):
        [_X_qs,_X_qu,_X_a,_X_b,_X_d] = _X
        _encoded_X = (jax.nn.softmax(_X_qs,0),jax.nn.softmax(_X_qu,0),tree_map(lambda a:a*a,_X_a),_X_b*_X_b,_X_d*_X_d)
        return _encoded_X
    
    
    # The actual function we'll minimize is this one :
    reduced_vfe = (lambda obs,_s,_u,a,b,d : act_hmm_free_energy(*((obs,) + ( _s,_u,a,b,d) + fixed_simulation_params)))
    
    # The loss function is the vfe !
    # Let's express it as a function of the observations and the vector of model parameters !
    loss_function = (lambda params,obs : reduced_vfe(*((obs,)+encode_params(params))))
    
    start_learning_rate = 1e-1
    optimizer = optax.adam(start_learning_rate)

    # Initialize parameters of the model + optimizer.

    params,loss_hist = fit(X,os,loss_function, optimizer,128)
    
    
    
    
    [post_qs,post_qu,post_a,post_b,post_d] = encode_params(params)
    
    def round(value):
        return np.round(np.array(value),2)
    print(round(trueA[0]))
    print(round(post_a[0]))
    
    print(round(trueA[1]))
    print(round(post_a[1]))
    
    for t in range(T):
        print(round(post_qu[:,t]),true_u[t])
    
    plt.plot(np.arange(0,len(loss_hist),1),loss_hist)
    plt.xlabel("Gradient descent steps")
    plt.ylabel("Variational Free Energy (logits)")
    plt.show()