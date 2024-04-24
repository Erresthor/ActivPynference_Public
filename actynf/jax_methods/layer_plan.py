import jax.numpy as jnp
import jax.random as jr
import jax

from numpyro import plate,sample,deterministic
import numpyro.distributions as distr

from jax.tree_util import tree_map
from jax import vmap
from jax import jit

from functools import partial
from itertools import product

from jax_toolbox import _normalize,_jaxlog
from planning_tools import compute_Gt_array,compute_novelty
from actynf.jax_methods.layer_infer_state import compute_state_posterior

# A set of functions for agents to plan their next moves.  Note that we use a bruteforce treesearch approach, which is
# a computational nightmare, but may be adapted to fit short term path planning.

@jit
def compute_G_action(action_vect,qs_tminus,
                 A,B,C,
                 A_novel,B_novel):
    # At a given timestep t_exp = t + i, with i in [0,Th[
    # what is the effect of action action_idx ?
    B_pi_t = B@action_vect # jnp.einsum("ijk,k->ij",B,qpi)
    
    qs_pi_tplus,_ = _normalize(B_pi_t @ qs_tminus) # Useless norm ?

    # qo is the list of predicted observation distributions at this time t given this qs_pi_tplus !
    qo = tree_map(lambda a_m : a_m @ qs_pi_tplus,A)

    Gt = compute_Gt_array(qo,qs_pi_tplus,qs_tminus,action_vect,
                          A,A_novel,B,B_novel,C)
    return Gt,qs_pi_tplus

@partial(jit, static_argnames=['Np'])
def compute_G_branch(qs_tminus,
                 A,B,C,
                 A_novel,B_novel,Np):
    one_action_G = lambda x : compute_G_action(x,qs_tminus,A,B,C,A_novel,B_novel)

    all_Gs,all_qsnext = vmap(one_action_G)(jnp.arange(Np))
    # Arrays of EFE at timestep t depending on action if no further timestep

    return all_Gs,all_qsnext

@partial(jit, static_argnames=['Np','Th'])
def scan_G_policy(policy_sequence,Th,Np,
                  qs_init,
                  A,B,C,
                  A_novel,B_novel):
    def _scanner(carry,t):
        qs = carry

        action_vector = jax.nn.one_hot(policy_sequence[t],Np)
        
        Gt,qs_next = compute_G_action(action_vector,qs,
                 A,B,C,
                 A_novel,B_novel)

        return qs_next,(qs,Gt)
    
    qs_horizon,(qss,complete_G_array) = jax.lax.scan(_scanner, qs_init, jnp.arange(0,Th,1))

    return complete_G_array,qss

@partial(jit, static_argnames=['Np','Th'])
def bruteforce_treesearch(Th,
                qs_init,
                A,B,C,
                A_novel,B_novel,Np):
    """ There MUST be a better way to do this, but it will do for now."""    
    scan_seq =  partial(scan_G_policy,Th = Th-1,Np=Np,qs_init=qs_init,A=A,B=B,C=C,A_novel=A_novel,B_novel=B_novel)
    actions_explored = jnp.arange(Np)
    all_combinations = jnp.array(jnp.meshgrid(*[actions_explored]*Th)).T.reshape(-1,Th)
    
    def treesearch_action_i(i):
        n_branches = all_combinations.shape[0] # Size Np**Th

        full_i = jnp.full((n_branches,1),i) # The first action analyzed is i
                                            # The subsequent ones are all the combinations
        sequences_analyzed = jnp.concatenate([full_i,all_combinations],axis=-1) 

        Gs_i,qss_i = vmap(scan_seq)(sequences_analyzed)
        return Gs_i,qss_i
    
    return vmap(treesearch_action_i)(jnp.arange(Np))
 
def nested_treesearch(cnt,
                qs_init,
                A,B,C,
                A_novel,B_novel):
    """ 
    TODO : There MUST be a better way to do this !
    """
    raise NotImplementedError("Nested treesearch is not implemented yet !")

@partial(jit, static_argnames=['Np','Th'])
def compute_EFE(Th,qs,A,B,C,A_novel,B_novel,Np):
    """ 
    lambda s -> G(u,s) for all allowable u
    """
    Gs,x = bruteforce_treesearch(Th,qs,A,B,C,A_novel,B_novel,Np)     

    Gs_compressed = Gs.sum(axis=(-1))
        # Gs_compressed[i,j,k] is the EFE of the agent having followed the single trajectory [Action(t)=i x Actions(t+1 -> t+2 -> ... -> t+Th)=j], at time k
        # = G(pi,tau)
    Gs_paths = Gs_compressed.sum(axis=(-1))
        # Gs_paths[i,j] is the sum of EFE of the agent having followed the single trajectory [Action(t)=i x Actions(t+1 -> t+2 -> ... -> t+Th)=j]
        # = G(pi) for all the combinations of actions in the temporal horizon
    
        # Assuming that the path selected after action i will be infered using softmax(G), the expected free energy after selecting action i is approximately :
    Gs_paths_norm = (jax.nn.softmax(Gs_paths,axis=-1)*Gs_paths).sum(axis=-1)

    return Gs_paths_norm

### Compute log policy posteriors --------------------------------------------------------------------------------
@partial(jit, static_argnames=['Np','Th','gamma'])
def policy_posterior(Th,qs,A,B,C,E,A_novel,B_novel,
                     gamma,Np):
    EFE_per_action = compute_EFE(Th,qs,A,B,C,A_novel,B_novel,Np) #(negative EFE)
    if (gamma==None):
        return EFE_per_action,jax.nn.softmax(EFE_per_action)
    return EFE_per_action, jax.nn.softmax(gamma*EFE_per_action + _jaxlog(E))

# @partial(jit, static_argnames=['Np','Th','gamma'])
def policy_posterior_reduced(Th,qs,A,B,C,E,gamma,Np):
    A_novel = compute_novelty(A,True)
    B_novel = compute_novelty(B)
    return policy_posterior(Th,qs,A,B,C,E,A_novel,B_novel,gamma,Np)


### Sample an action from the posterior --------------------------------------------------------------------------
@jax.jit
def alpha_weight(raw_posterior,alpha):
    alpha_weighted_posterior = jax.nn.softmax(_jaxlog(raw_posterior) * alpha)
    return alpha_weighted_posterior

def sample_action(qpi,Np,alpha, selection_method="deterministic",rng_key=None):
    if selection_method == "deterministic":
        action_idx = jnp.argmax(qpi)
        action_dist = jax.nn.one_hot(action_idx,Np)
        action_vect = jax.nn.one_hot(action_idx,Np)
    elif selection_method == "stochastic":
        action_dist = alpha_weight(qpi,alpha)
        action_idx = jr.categorical(rng_key, _jaxlog(action_dist))
        action_vect = jax.nn.one_hot(action_idx,Np)
    return action_dist,action_idx,action_vect

def sample_action_pyro(qpi,Np,alpha,selection_method = "stochastic_raw",observed_action=None):
    """ 
    When lost about shapes in numpyro :
    https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/
    """
    if observed_action != None:
        assert qpi.shape[:-1]==observed_action.shape,"Shape mismatch in sample_action_pyro"
    
    if selection_method == "deterministic":
        action_idx = jnp.argmax(qpi,axis=-1)
        action_dist = jax.nn.one_hot(action_idx,Np,axis=-1)
        action_vect = jax.nn.one_hot(action_idx,Np,axis=-1)
        deterministic("action_t",action_idx)
    elif selection_method == "stochastic_alpha":
        action_dist = alpha_weight(qpi,alpha) # Along -1th axis
        action_idx = sample("actions",distr.Categorical(probs=action_dist).to_event(action_dist.ndim-1),obs=observed_action)
        action_vect = jax.nn.one_hot(action_idx,Np)
    elif selection_method == "stochastic_raw":
        action_dist = qpi
        action_idx = sample("actions",distr.Categorical(probs=action_dist).to_event(action_dist.ndim-1),obs=observed_action)
        action_vect = jax.nn.one_hot(action_idx,Np)
    return action_dist,action_idx,action_vect

if __name__ == '__main__':
    import random as ra
    import numpy as np
    Nos = np.array([10,10])
    Ns = 10
    T = 10
    Np = 10

    key = jr.PRNGKey(464)

    fixed_observations = [np.random.randint(0,No,(T,)) for No in Nos]
    

    # A = [_normalize(jr.uniform(key,(No,Ns)))[0] for No in Nos]

    Nmod = 2
    A = [_normalize(jnp.eye(Ns))[0] for i in range(Nmod)]

    C = [jnp.zeros((a.shape[0],)) for a in A]
    C[0] = jnp.linspace(0,11,C[1].shape[0])

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

    A_novel = compute_novelty(A,True)
    B_novel = compute_novelty(B)
    qsm,_ = _normalize(jr.uniform(key,(Ns,)))

    key,key2 = jr.split(key)
    qsp,_ = _normalize(jr.uniform(key2,(Ns,)))    

    E = jnp.ones((Np,))

    Th= 4
    gamma = None
    res = policy_posterior_reduced(Th,qsm,A,B,C,E,gamma,Np)
    print(res)
    exit()
    

    N_a_lot = 20
    key,key2 = jr.split(key)
    qs_along_a_lot_of_timesteps,_ = _normalize(jr.uniform(key2,(Ns,N_a_lot)),axis=0)
    
    
    print(qsm)
    def _scan(carry,X):
        
        s_prior,o = carry
        key,key_agent,key_process = jr.split(X,3)

        qs = compute_state_posterior(s_prior,o,A)

        _efe,_qpi,_u = sample_action(Th,Np,qs,A,B,C,E,A_novel,B_novel,alpha=2,gamma = None,selection_method="stochastic",rngkey=key_agent)

        new_qs = B[...,_u]@qs # jax.nn.one_hot(_u,Ns)

        new_o = tree_map(lambda A_m: A_m@new_qs,A)
        return (new_qs,new_o),(_qpi,new_qs,_u,new_o)
    
    rngkey = jr.PRNGKey(2335)
    next_keys = jr.split(rngkey, T - 1)

    print(next_keys)

    obs_vectors = [jax.nn.one_hot(rvs,No,axis=0) for rvs,No in zip(fixed_observations,Nos)]
    
    (last_s,last_o),(qpi_arr,s_arr,u_arr,o_arr) = jax.lax.scan(_scan, (qsm,[o_mod[:,0] for o_mod in obs_vectors]),next_keys)
    print(qpi_arr)
    print(np.round(np.array(s_arr),2))
    print(o_arr)
    print(u_arr)
    exit()
    
    for t in range(N_a_lot):
        key,key2 = jr.split(key)
        Q,qpi,u = infer_action(Th,qsm,A,B,C,E,A_novel,B_novel,alpha=2,gamma = None,selection_method="deterministic",rngkey=key2)
        print(Q)
        print(qpi)
        print(u)
        print("------------------")