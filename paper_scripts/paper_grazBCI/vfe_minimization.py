# -*- coding: utf-8 -*-
"""
A Jax implementation of the VFE of a flexible transitions HMM 
for gradient descent !
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
import jax.random as jr

from jax import vmap,jit
from jax.tree_util import tree_map
from jax.scipy.special import gammaln,digamma
from functools import partial

import optax

import matplotlib.pyplot as plt
import functools 

from actynf.jaxtynf.jax_toolbox import _jaxlog,_normalize


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

    return t1 + t2 + (t3*t4).sum()



def _kl_dir_dirichlet_list(list_1,list_2,epsilon=1e-5):
    """
        Computes the KL_dir between two lists of tensors of dirichlet parameters along axis 0.
        In : 
        - list_1 and list_2 are both lists of same length, which elements are arrays of same shape of size No x Np_1 x Np_2 x ... x Np_nmod
        
        We reshape each tensor array to be of shape No x K where K is a big number :D (Np_1 x Np_2 x ... x Np_nmod)
        
        Returns : 
        - A tensor of shape len(list_1)=len(list_2) where each element is the KL_dir of the respective element couple
    """
    def _kl_dir_one_mod(list_1_mod,list_2_mod):
        N_categories = list_1_mod.shape[0]
        assert N_categories == list_2_mod.shape[0], "Mismatch in Dirichlet Distribution KL dir computation"
        
        # We got to ensure that this element is 2 dimensionnal, with the leading dimension being the one the
        # dirichlet KL divergence will be computed against :
        reshaped_list_1_mod = jnp.reshape(list_1_mod,(N_categories,-1))
        reshaped_list_2_mod = jnp.reshape(list_2_mod,(N_categories,-1))

        kl_two_dists = _kl_dir_dirichlet(reshaped_list_1_mod,reshaped_list_2_mod,epsilon)

        return jnp.sum(kl_two_dists)
    
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



def emission_term_one_factor(o,qs,logA):
    """Compute the emission term of the VFE likelihood for
    the FTHMM. (for 1dimensional latent states only)
    """
    def emission_one_mod(o_m,logA_m):
        return jnp.einsum("ti,ji,tj->t",qs,logA_m,o_m)
    accuracy_all_mods = tree_map(emission_one_mod,o,logA)
    return jnp.stack(accuracy_all_mods).sum(axis=0)

def emission_term_multiple_factors(o,qs,logA,o_filter=None):
    """Compute the emission term of the VFE likelihood for
    the FTHMM. 

    Args:
        o (_type_): a list of Nmod tensor arrays of shape [Nout[mod]] encoding the emissions.
        qs (_type_):  a list of Nf tensor arrays of shape [Ns[f]] encoding the hiddens states.
        logA (_type_): a list of Nmod tensor arrays of shape [Nout[mod] x Ns(1) x Ns(2) x ... x Ns(Nf)] encoding emissions depedning on the hidden states.
    """
    # Reshape mods :
    _N_outcomes = len(logA)
    _latent_state_tuple = (1,)*(logA[0].ndim-1)
    
    if o_filter is None:
        o_filter = jnp.ones((_N_outcomes,))
    o_filter = jnp.reshape(o_filter,(_N_outcomes,)+_latent_state_tuple)
   
   
    def emission_term_one_mod(_logA_mod,_o_mod):    
        _Noutcomes_mod = _o_mod.shape[0]
        
        # reshape o to fit the matrix : 
        _reshaped_o_mod = jnp.reshape(_o_mod,(_Noutcomes_mod,)+_latent_state_tuple)
            
        joint_emission_logA = jnp.sum(_reshaped_o_mod*_logA_mod,axis=0)
        return joint_emission_logA
    
    # From pymdp (compute_accuracy method):
    x = qs[0]
    for q in qs[1:]:
        x = jnp.expand_dims(x, -1) * q
    
    # return tree_map(emission_term_one_mod,logA,o)
    log_all_modalities = jnp.stack(tree_map(emission_term_one_mod,logA,o))
    
    # We filter out the unseen observations here !
    filtered_log_modalities = (o_filter*log_all_modalities).sum(axis=0)
    return filtered_log_modalities*x
    
    

# Those terms use the full state posterior directly, no need to map them across timesteps :
def transition_term(qs,logB,qu,vecU):
    
    # We need a term here that translates action posteriors (1-dim)
    # to state_transitions (1 per factor):
    
    
    def transition_term_one_fac(_qs_f,_logB_f,_transition_mapping_f):
        _qs_tminus_f = _qs_f[-1:]
        _qs_tplus_f = _qs_f[1:]
        
        _transition_f = jnp.einsum("ks,tk->ts",_transition_mapping_f,qu)
        return jnp.einsum("ti,ijk,tj,tk->t",_qs_tplus_f,_logB_f,_qs_tminus_f,_transition_f)
    
    transition_log_prob = tree_map(transition_term_one_fac,qs,logB,vecU)
    return jnp.stack(transition_log_prob).sum(axis=0)
    
def initial_term(qs,logD):
    def initial_term_one_fac(_qs_f,_logD_f):
        _qs_init_f = _qs_f[0,:]
        return jnp.einsum("i,i->",_qs_init_f,_logD_f)
    return jnp.stack(tree_map(initial_term_one_fac,qs,logD)).sum(axis=0)

# This term accepts only data from a given timepoint, need to map it across its 3 leading dims :
@partial(vmap, in_axes=(0,0,0,None))
def emission_term_multiple_factors(o,o_filter,qs,logA):
    """Compute the emission term of the VFE likelihood for
    the FTHMM. 

    Args:
        o (_type_): a list of Nmod tensor arrays of shape [Nout[mod]] encoding the emissions.
        qs (_type_):  a list of Nf tensor arrays of shape [Ns[f]] encoding the hiddens states.
        logA (_type_): a list of Nmod tensor arrays of shape [Nout[mod] x Ns(1) x Ns(2) x ... x Ns(Nf)] encoding emissions depedning on the hidden states.
    """
    # Reshape mods :
    _N_outcomes = len(logA)
    _latent_state_tuple = (1,)*(logA[0].ndim-1)
    
    def emission_term_one_mod(_logA_mod,_o_mod):    
        _Noutcomes_mod = _o_mod.shape[0]
        
        # reshape o to fit the matrix : 
        _reshaped_o_mod = jnp.reshape(_o_mod,(_Noutcomes_mod,)+_latent_state_tuple)
            
        joint_emission_logA = jnp.sum(_reshaped_o_mod*_logA_mod,axis=0)
        return joint_emission_logA
    
    # From pymdp (compute_accuracy method):
    x = qs[0]
    for q in qs[1:]:
        x = jnp.expand_dims(x, -1) * q
    
    # return tree_map(emission_term_one_mod,logA,o)
    log_all_modalities = jnp.stack(tree_map(emission_term_one_mod,logA,o))
    
    # We filter out the unseen observations here !
    o_filter_reshaped = jnp.reshape(o_filter,(_N_outcomes,)+_latent_state_tuple)
    filtered_log_modalities = (o_filter_reshaped*log_all_modalities).sum(axis=0)
    return filtered_log_modalities*x


# This function accepts only posteriors from a given timepoint, need to map it across its leading dim
@partial(vmap)
def posterior_entropy(qs):
    def posterior_entropy_factor(_qs_f):
        assert _qs_f.ndim==1, "Dimension problem in entropy computation :'("
        return (_qs_f*_jaxlog(_qs_f)).sum()
    return jnp.stack(tree_map(posterior_entropy_factor,qs))  # Entropies for all state factors

@jit
def act_hmm_free_energy(observations, observations_filter,
                        qs ,qu,
                        A, B, D, 
                        pu,pA,pB,pD,U,
                        epsilon_kl=1e-5):
    """Compute the variational free energy from a HMM given a set of observations and priors.
    This HMM is a 'FTHMM' (Flexible Transitions HMM) and we consider that there exist multiple 
    possible transitions that could explain hidden state transitions. Importantly, these transitions are
    not directly observed ...
    This may be a hard model to fit, but may work well with well defined priors.

    Args:
        observations (_type_): A list of Nmod jax tensor arrays. Each of these array should be of shape (Ntimesteps x Noutcomes[modality])
        observations_filter (_type_): A single (Ntimesteps x Nmod) jax tensor array of binaries encoding wether or not a specific observation modality was seen at this timepoint.
        qs (_type_): A list of Nf jax tensor arrays of hidden state posteriors. Each of these array should be of shape (Ntimesteps x Nstates[factor])
        qu (_type_): A single (Ntimesteps-1 x Nactions) jax tensor array of action posteriors.
        A (_type_): A list of Nmod jax tensor arrays encoding the Dirichlet parameter vector for the posterior
                    emission rules for this HMM. Each of these arrays should be of shape (Noutcomes[modality] x Ns[factor1] x ... x Ns[factorN])
        B (_type_): A list of Nf jax tensor arrays encoding the Dirichlet parameter vector for the posterior 
                    transition rules for this HMM. Each of these arrays should be of shape (Ns[factor] x Ns[factor] x Ntransitions[factor])
                    (note the dependency on the transitions.)
        D (_type_): A list of Nf jax tensor arrays encoding the Dirichlet parameter vector for the posterior 
                    initial state rules for this HMM. Each of these arrays should be of shape (Ns[factor])
        pu (_type_): _description_
        pA (_type_): A list of Nmod jax tensor arrays encoding the Dirichlet parameter vector for the prior
                    emission rules for this HMM. Each of these arrays should be of shape (Noutcomes[modality] x Ns[factor1] x ... x Ns[factorN])
        pB (_type_): A list of Nf jax tensor arrays encoding the Dirichlet parameter vector for the prior 
                    transition rules for this HMM. Each of these arrays should be of shape (Ns[factor] x Ns[factor] x Ntransitions[factor])
                    (note the dependency on the transitions.)
        pD (_type_): A list of Nf jax tensor arrays encoding the Dirichlet parameter vector for the prior 
                    initial state rules for this HMM. Each of these arrays should be of shape (Ns[factor])
        U (_type_): A single (Nactions x Nfactors) jax tensor array encoding the mapping between allowable actions
                    and factor-wise transitions.
        epsilon_kl (_type_, optional): a small value to avoid numerical over/under-flows. Defaults to 1e-5.

    Returns:
        vfe (scalar): The computed VFE of the model given the input data
    """
    # the free energy for a simple hmm model with flexible transitions !
    # o,A and pA are lists of tensors with as many elements as there are observation modalities
    # qs, B, D , pB and pD are lists of tensors with as many elements as there are state tensors
    # qu and pu are 2 dimensional tensors of (Ntimesteps x Nallowables actions)
    # Actions need to be related to factorwise transitions through the U (static) tensor ()
    
    # This function is defined at the **trial** scale, with qs,qu and o having Ntimesteps as their leading dimension !
    
    
    N_actions,N_factors = U.shape
    N_transitions = [b_f.shape[-1] for b_f in B]
    # /!\ N_transitions != N_actions : 
    # - Transitions are defined for each hidden state factor
    # - Actions are defined at the model level. Only one action can occur each timestep. 
    # The input U provides a direct mapping between actions and transitions.
    
    
    # Because in a multifactorial setting, actions != transitions,
    # we need to define an action -> factor transition mapping. 
    # This is a static value (does not evolve during the model inversion)
    vecU = [jax.nn.one_hot(U[:,factor],N_transitions[factor]) for factor in range(N_factors)]

    # 1. NETWORK PARAMETER DIVERGENCES !
    # Parameter KL divergences with their respective priors
    kl_a = _kl_dir_dirichlet_list(A,pA,epsilon_kl)
    kl_b = _kl_dir_dirichlet_list(B,pB,epsilon_kl)
    kl_d = _kl_dir_dirichlet_list(D,pD,epsilon_kl)

    # 2. HIDDEN VARIABLES (STATE + ACTION) POSTERIOR ENTROPIES
    # This should be computed across all state factors :
    # neg_Hs = (qs*_jaxlog(qs)).sum(axis=0)   # State entropies over all timesteps / summed over all states
    
    neg_Hu = (qu*_jaxlog(qu)).sum(axis=0)  # Action entropies over all timesteps / summed over all allowables
    neg_Hs = posterior_entropy(qs)
        # [Ntimesteps x Nfactors]

    # Term I : parameter prior divergences + posterior entropies
    term1 = kl_a.sum() + kl_b.sum() + kl_d.sum() + neg_Hs.sum() + neg_Hu.sum()


    # 3. PRIOR CROSS ENTROPIES 
    # (actions only, as state priors are parametrized by B and D)
    negHpu = (qu*_jaxlog(pu)).sum(axis=-1)  # Negative cross entropy between qu and prior pu:

    # 4. LIKELIHOODS !
    # Compute the log expectations of the parameters under their 
    # respective Dirichlet distribution once only :
    logA = _logexpect_dirichlet_list(A) # A list of logA for each observation modality
    logB = _logexpect_dirichlet_list(B) # A list of logB for each state factor
    logD = _logexpect_dirichlet_list(D) # A list of logD for each state factor

    log_initial = initial_term(qs,logD)
    log_transition = transition_term(qs,logB,qu,vecU)
    log_perception = emission_term_multiple_factors(observations,observations_filter,qs,logA)
        
    # TERM II : (accuracy)
    term2 = log_initial + log_transition.sum() + log_perception.sum() + negHpu.sum()
    vfe = term1 - term2
    return vfe


# SGD on the VFE for parameter inference
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




def test_vfe_computation():
    # Ntrials = 10
    Ntimesteps = 20
    
    rng_keys  =[0,1,2,3,4]
    Noutcomes = [10,5,6,4,8]
    Nmodalities = len(Noutcomes)
    
    Ns = (5,4)
    Nfactors = len(Ns)
    
    obs_indexes = [jr.randint(jr.PRNGKey(key),(Ntimesteps,),0,nout) for (key,nout) in zip(rng_keys,Noutcomes)]
    obs_vect = tree_map((lambda a,b: jax.nn.one_hot(a,b)),obs_indexes,Noutcomes)
    obs_filter = jnp.ones((Ntimesteps,Nmodalities))
    
    qs = [_normalize(jnp.ones((Ntimesteps,Ns_f)),axis=-1)[0] for Ns_f in Ns]
    
    # HMM weights :
    A = [jnp.ones((nout,) + Ns) / 3.0 for nout in Noutcomes]
    
    B = [jnp.ones((5,5,4)),jnp.ones((4,4,2))] # Factor transitions
    
    D = [jnp.ones((5,)),jnp.ones((4,))]
    
    # Action - transition mapping 
    U = jnp.array([
        [1,0],
        [0,0],
        [3,1],
        [2,1]
    ])
    
    Nactions,Nfactors = U.shape
    
    
    
    
    print([o.shape for o in obs_vect])
    print([s.shape for s in qs])
    
    
    qu,_ = _normalize(jnp.ones((Ntimesteps-1,Nactions)),-1)
    pu,_ = _normalize(jnp.ones((Ntimesteps-1,Nactions)),-1)
     
    
    pA = A
    pB = B
    pD = D
    
    trial_vfe = act_hmm_free_energy(obs_vect, obs_filter,
                        qs ,qu,
                        A, B, D, U,
                        pu,pA,pB,pD,
                        epsilon_kl=1e-5)
    
    print(trial_vfe)


if __name__ == '__main__':
    T = 2
    Ns = 5
    No1 = 5
    No2 = 10
    Nu = 10
    
    Noutcomes = [No1,No2]
    Nmodalities = len(Noutcomes)
    rng_keys = [10,15]
    
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

    qs,_ = _normalize(jnp.array(np.random.randint(1,10,(T,Ns))))
    qu,_ = _normalize(jnp.array(np.random.randint(1,10,(T-1,Nu))))

    pu,_ = _normalize(jnp.ones_like(qu))
    
    U = jnp.expand_dims(jnp.arange(Nu),-1)
    print(U)
    key = jax.random.PRNGKey(9999)
    trueA,trueB,trueD = [_normalize(a)[0] for a in A],_normalize(B)[0],_normalize(D)[0]
    
    obs_indexes = [jr.randint(jr.PRNGKey(key),(T,),0,nout) for (key,nout) in zip(rng_keys,Noutcomes)]
    obs_vect = tree_map((lambda a,b: jax.nn.one_hot(a,b)),obs_indexes,Noutcomes)
    obs_filter = jnp.ones((T,Nmodalities))
    

    # Network parameters : 
    
    
    # The initial parameters are a set of tensors  (should we randomize them ?)
    # Initial variational posteriors
    X_qs = jnp.ones_like(qs) 
    X_qu = jnp.ones_like(qu)
    # Initial hmm parameters
    X_a = [jnp.ones_like(a) for a in A]
    X_b = [jnp.ones_like(B)]
    X_d = [jnp.ones_like(D)]
    
    X = [X_qs,X_qu,X_a,X_b,X_d]

    # That we encode to make them acceptable parameters for our HMM model
    def encode_params(_X):
        [_X_qs,_X_qu,_X_a,_X_b,_X_d] = _X
        
        xsquared = lambda a : a*a
        _encoded_X = ([jax.nn.softmax(_X_qs,-1)],jax.nn.softmax(_X_qu,-1),tree_map(xsquared,_X_a),tree_map(xsquared,_X_b),tree_map(xsquared,_X_d))
        return _encoded_X
    
    
    # The actual function we'll minimize is this one :
    # We assume that our priors are fixed
    fixed_simulation_params = (pu,pA,[pB],[pD],U)
    # And here is the function depending only on the data and our parameters of interest
    def fthmm_wrapper(data,_s,_u,_a,_b,_d):
        _obs_vec, _obs_filter = data
        return act_hmm_free_energy(*((_obs_vec,_obs_filter) + (_s,_u,_a,_b,_d) + fixed_simulation_params))
    # reduced_vfe = (lambda obs,_s,_u,a,b,d : act_hmm_free_energy(*((obs,) + ( _s,_u,a,[b],[d]) + fixed_simulation_params)))
    reduced_vfe = fthmm_wrapper
    
    # The loss function is the composition of the reduced vfe and our encoding function :
    # Let's express it as a function of the observations and the vector of model parameters !
    loss_function = (lambda params,data : reduced_vfe(*((data,)+encode_params(params))))
    
    start_learning_rate = 1e-1
    optimizer = optax.adam(start_learning_rate)

    # Initialize parameters of the model + optimizer.
    data = (obs_vect,obs_filter)
    
    params,loss_hist = fit(X,data,loss_function, optimizer,128)
    
    
    
    
    [post_qs,post_qu,post_a,post_b,post_d] = encode_params(params)
    
    def round(value):
        return np.round(np.array(value),2)
    # print(round(trueA[0]))
    print(round(post_a[0]))
    
    # print(round(trueA[1]))
    print(round(post_a[1]))
    
    for t in range(T):
        print(round(post_qu[t,:]))
    
    plt.plot(np.arange(0,len(loss_hist),1),loss_hist)
    plt.xlabel("Gradient descent steps")
    plt.ylabel("Variational Free Energy (logits)")
    plt.show()