import jax.numpy as jnp
import jax.random as jr
import jax

from numpyro import plate,sample,deterministic
import numpyro.distributions as distr

from jax.tree_util import tree_map
from jax import vmap
from jax import jit

from functools import partial

from .jax_toolbox import _normalize,_jaxlog
from .planning_tools import compute_Gt_array,compute_novelty
from .layer_infer_state import compute_state_posterior

# A set of functions for agents to plan their next moves.  Note that we use a bruteforce treesearch approach, which is
# a computational nightmare (especially for high temporal horizons), but may be adapted to fit short term path planning.

@partial(jit,static_argnames=["efe_a_nov","efe_b_nov","efe_old_a_nov"])
def compute_G_action(t,
            action_vect,qs_tminus,
            A,B,C,
            A_novel,B_novel,
            efe_a_nov=True,efe_b_nov=False,efe_old_a_nov=False):
    """Compute the expected free energy of a specific action vector performed at time t
    given a context prior qs_tminus and a generative model A,B,C

    Args:
        t (_type_): _description_
        action_vect (_type_): _description_
        qs_tminus (_type_): _description_
        A (_type_): _description_
        B (_type_): _description_
        C (_type_): _description_
        A_novel (_type_): _description_
        B_novel (_type_): _description_

    Returns:
        _type_: _description_
    """
    # At a given timestep t_exp = t + i, with i in [0,Th[
    # what is the effect of action action_idx ?
    qs_pi_tplus,_ = _normalize(jnp.einsum('iju,j,u->i',B,qs_tminus,action_vect)) # Useless norm ?

    # qo is the list of predicted observation distributions at this time t given this qs_pi_tplus !
    # a_m @ qs_pi_tplus
    qo = tree_map(lambda a_m : jnp.einsum('oi,i->o',a_m,qs_pi_tplus),A)

    Gt = compute_Gt_array(t,qo,qs_pi_tplus,qs_tminus,action_vect,
                          A,A_novel,B,B_novel,C,
                          efe_a_nov,efe_b_nov,efe_old_a_nov)
    return Gt,qs_pi_tplus

@partial(jit, static_argnames=['efe_a_nov',"efe_b_nov","efe_old_a_nov"])
def scan_G_policy(policy_sequence,
                  initial_t,
                  qs_init,
                  A,B,C,E,
                  A_novel,B_novel,
                efe_a_nov=True,efe_b_nov=False,efe_old_a_nov=False):
    Th = policy_sequence.shape[0]
    Np = B.shape[-1]
    
    def _scanner(carry,ti):
        qs = carry

        action_vector = jax.nn.one_hot(policy_sequence[ti],Np)
        
        habits_biais = E[policy_sequence[ti]]
        
        Gt,qs_next = compute_G_action(initial_t + ti, # This is the timestep of the action
                 action_vector,qs,
                 A,B,C,
                 A_novel,B_novel,
                 efe_a_nov,efe_b_nov,efe_old_a_nov)

        return qs_next,(qs_next,Gt,habits_biais)
    
    qs_horizon,(qss,complete_G_array,habits_biaises) = jax.lax.scan(_scanner, qs_init, jnp.arange(0,Th,1))

    return complete_G_array,habits_biaises,qss

@partial(jit, static_argnames=['Th','efe_a_nov','efe_b_nov','efe_old_a_nov'])
def bruteforce_treesearch(initial_t,Th,
                qs_init,
                A,B,C,E,
                A_novel,B_novel,
                filter_end_of_trial,
                efe_a_nov=True,efe_b_nov=False,efe_old_a_nov=False):
    """ 
    Computes the EFE for the various possible policy paths.
    
    This method does NOT create separate branches for each plausible hidden state. Instead,
    it uses the prior over the subsequent state given the action directly. This is not 
    sufficient when trying to resolve uncertainty.
    
    """  
    assert Th>=2,"Temporal horizon for planning should be >=2"
    
    Ns = qs_init.shape[0]
    Np = B.shape[-1]
    actions_explored = jnp.arange(Np)
      
    scan_seq =  partial(scan_G_policy,initial_t=initial_t,qs_init=qs_init,A=A,B=B,C=C,E=E,A_novel=A_novel,B_novel=B_novel,efe_a_nov=efe_a_nov,efe_b_nov=efe_b_nov,efe_old_a_nov=efe_old_a_nov)
            # This is a fixed policy tree that the agent will explore. This is based on static parameters (Th and Np)
    
    combinations_idxs = [actions_explored]*Th
    all_combinations = jnp.stack(jnp.meshgrid(*combinations_idxs,indexing="ij"),axis=-1).reshape(-1,Th)
            # All possible action combinations from initial_t to initial_t + Th
            # (arguably a very highly-sized tensor ^^)
        
    gs_all_i,habs_all_i,qss_all_i = vmap(scan_seq)(all_combinations)
        # gs_all_i[actions_explored**Th,Th,4] --> the last component is a decomposition of each EFE term
    
    # compressed_last_dimension=  gs_all_i.sum(axis=-1) + habs_all_i
    compressed_last_dimension=  gs_all_i.sum(axis=-1)
        # We don't care about each component of the EFE for non-diagnostic purposes
    
    end_of_trial_filtered = jnp.einsum("ij,j->ij",compressed_last_dimension,filter_end_of_trial) + habs_all_i
        # This is a (Np**(Th)) * Th tensor
        # Let's unfold the (Np**(Th)) into (Np*Np*...*Np) [Th dimensions]
    

    def unfold_dims(tensor,wanted_shape):
        return tensor.reshape(*wanted_shape)
    
    temporal_tree_shape = [Np]*(Th) + [Th]
    efes_vectorized = unfold_dims(end_of_trial_filtered,temporal_tree_shape)
    # return end_of_trial_filtered,efes_vectorized
    state_temporal_tree_shape = [Np]*(Th) + [Th,Ns]
    states_vectorized = unfold_dims(qss_all_i,state_temporal_tree_shape)

    # Return the unfolded matrices : 
    return  efes_vectorized, states_vectorized, end_of_trial_filtered,qss_all_i

@partial(jit, static_argnames=['Th','efe_compute_a_nov','efe_compute_b_nov','old_a_nov'])
def compute_EFE(t,Th,filter_end_of_trial,
                qs,A,B,C,E,A_novel,B_novel,
                efe_compute_a_nov,efe_compute_b_nov,old_a_nov):
    """ 
    lambda s -> G(u,s) for all allowable u
    
    To be compared with spm_forwards !    
    """
        
    efe_compute_a_nov = False
    efe_compute_b_nov = False
    vect_efe,vect_states, flat_efe,flat_states = bruteforce_treesearch(t,Th,
                                                qs,
                                                A,B,C,E,
                                                A_novel,B_novel,filter_end_of_trial,
                                                efe_a_nov=efe_compute_a_nov,
                                                efe_b_nov=efe_compute_b_nov,
                                                efe_old_a_nov=old_a_nov)
            # For each path of actions i1->i2->...->iTh, get the efe at each timestep
            # vect efe is a Np x Np x Np x ... x Np x Th tensor
    
    # _________________________________________________________________________________
    # Collapse ! We've moved forward in the tree, let's move backwards and perform successive summations 
    # of EFEs given the predictive action posterior!
    # _________________________________________________________________________________
    updating_efe_tree = jnp.copy(vect_efe)
    updating_state_tree = jnp.copy(vect_states)
        # Every iteration, we want to "collapse" thes tree along their last temporal horizon dimension

    # expected_fe_next_step = jnp.zeros_like(updating_efe_tree[...,0])      
         # The agent does not predicts the EFE for timesteps > t + Th
         # Instead, it uses its habits ! 
    # expected_fe_last_step is a Np x Np x ... x Np array
    # Where each last cell has the same vecE*softmax(vecE)
    expected_fe_last_step = jnp.full_like(updating_efe_tree[...,0],jnp.sum(E*jax.nn.softmax(E)))
    expected_fe_next_step = expected_fe_last_step
    
    
    # This will be unrolled ! (needs to be done sequentially, 
    # big Th values are obviously discouraged)
    # Autobots, roll out !
    for index in range(-1,-Th,-1):
        # if index == -Th+1:
        #     return updating_efe_tree[...,index]
        intermediate_value = updating_efe_tree[...,-1] + expected_fe_next_step
        # if index == -4:
        #     return [intermediate_value.flatten()]
        softmax_for_index = jax.nn.softmax(intermediate_value,axis=(-1))
                    # Expected action at time Th+index
        
        expected_fe_next_step = (softmax_for_index*intermediate_value).sum(axis=-1)       
                    # A Np x ... x Np (Th - index elements) expected value tensor
                    # The values are the same along the index-th dimension
                    # Note that the elements on dimensions > index are irrelevant 
                    # to the remainder of the computations

        # Predictive posterior over hidden states, store this somewhere if needed
        pred_state_posterior = (jnp.expand_dims(softmax_for_index,-1)*updating_state_tree[...,-1,:]).sum(axis=-2)
                        # = Expectation of the hidden state at t + Th + index given the action posterior

        # Pruning the exploratoration trees, this should not be done for the very last summation
        # At index, all the values in updating_tree are the same 
        updating_efe_tree = updating_efe_tree[...,0,:-1] # We don't need this dimension any longer !
        updating_state_tree = updating_state_tree[...,0,:-1,:] # We don't need this dimension any longer !

    last_G =  updating_efe_tree[...,0] + expected_fe_next_step
    # return [last_G]
    
    last_action_posterior = jax.nn.softmax(last_G)
    last_pred_posterior = (jnp.expand_dims(last_action_posterior,-1)*updating_state_tree[...,0,:]).sum(axis=-2)
    return last_G,last_action_posterior,last_pred_posterior

### Compute log policy posteriors --------------------------------------------------------------------------------
# @partial(jit, static_argnames=['Np','Th','gamma'])
def policy_posterior(current_timestep,Th,filter_end_of_trial,
                     qs,A,B,C,E,A_novel,B_novel,
                     planning_options):
    Np = B.shape[-1]
    
    efe_compute_a_nov = planning_options["a_novelty"]
    efe_compute_b_nov = planning_options["b_novelty"]
    old_efe_computation = planning_options["old_novelty_computation"]
    
    EFE_per_action,last_action_posterior,predictive_state_posterior = compute_EFE(current_timestep,Th,filter_end_of_trial,
                                                        qs,
                                                        A,B,C,E,A_novel,B_novel,
                                                        efe_compute_a_nov,efe_compute_b_nov,old_efe_computation) #(negative EFE)
    return EFE_per_action,last_action_posterior
    # return EFE_per_action, jax.nn.softmax(gamma*EFE_per_action + _jaxlog(E))

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