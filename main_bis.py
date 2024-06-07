import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax,vmap, jit
from jax.tree_util import tree_map
from functools import partial,reduce


import actynf
print("Actynf version : " + str(actynf.__version__))
from actynf.jax_methods.layer_training import synthetic_training_multi_subj
from actynf.jax_methods.layer_options import get_planning_options
from actynf.jax_methods.planning_tools import autoexpand_preference_matrix

from actynf.jax_methods.jax_toolbox import _normalize

from actynf.jax_methods.planning_tools import compute_novelty

from demos.jax_demos.utils.basic_task import build_training_process,build_subject_model

from actynf.jax_methods.layer_trial import synthetic_trial,synthetic_trial_set_vals,synthetic_trial_direct_run,compute_step_posteriors,end_of_trial_filter

from actynf.jax_methods.layer_process import fetch_outcome,initial_state_and_obs

from actynf.demo_tools.tmaze.weights import get_T_maze_gen_process,get_T_maze_model
# def kronecker_prod_action(b_normed):
    
from actynf.jax_methods.layer_infer_state import compute_state_posterior
from actynf.jax_methods.layer_learn import learn_after_trial,backwards_pass
from actynf.jax_methods.jax_toolbox import _normalize,convert_to_one_hot_list,_swapaxes
from actynf.jax_methods.layer_options import DEFAULT_PLANNING_OPTIONS,DEFAULT_LEARNING_OPTIONS,get_learning_options

from actynf.jax_methods.layer_infer_state import compute_state_posterior
from actynf.jax_methods.shape_tools import to_log_space,vectorize_weights,get_vectorized_novelty

from actynf.jax_methods.layer_plan_tree import bruteforce_treesearch,compute_Gt_array


def example():
    A0 = jnp.array([
        [0.9,0.1],
        [0.1,0.9]
    ])
    A1 = jnp.array([
        [0.1,0.9],
        [0.9,0.1]
    ])
    A = [A0,A1]
    
    B0 = jnp.array([
        [0.1,0.9],
        [0.9,0.1]
    ])
    B1 = jnp.array([
        [0.9,0.1],
        [0.1,0.9]
    ])
    B = jnp.stack([B0,B1],-1)
    # print(B[...,0])
    # print(B[...,1])
    # exit()
    
    D = jnp.array([0.1,0.9])
    
    C = [jnp.array([0.0,0.0]),jnp.array([0.0,10.0])]
    # for mod in range(len(C)):
    #     C[mod] = jnp.expand_dims(C[mod],)
    
    
    s = jnp.array([0.8,0.2])
    
    key = jax.random.PRNGKey(6666)

    # res = fetch_outcome(key,2,[2,2],
    #         0,jnp.array([0.6,0.4]),jnp.array([1,0]),
    #         A,B,D,
    #         fixed_states_array=None,fixed_outcomes_tree=[None,None])

    # print(res)
    
    Np = B.shape[-1]
    
    
    
    static_states = jnp.array([0,-1,-1,-1,0,1,-1,-1,-1,1])
    
    qss,qpis,efes,states,obs = synthetic_trial_set_vals(key,
              2,[2,2],Np,
              [10*A0,10*A1],10*B,C,D,jnp.ones((Np,)),
              A,B,D,
              static_set_states = static_states,static_set_obs = None,
              T=10,Th =3,
              alpha = 16,gamma = None, selection_method="stochastic")
    
    print(qpis)
    print(qss)

def run_training_multi_factors(rngkey,Ns,Nos,Np,
            Ntrials,T,
            inA,inB,inD,inU,
            a0,b0,c,d0,e,
            Th =3,
            selection_method="stochastic",alpha = 16,gamma = None, 
            planning_options=DEFAULT_PLANNING_OPTIONS,
            learn_dictionnary = DEFAULT_LEARNING_OPTIONS):
    """
    Contrary to previous methods, this accepts multiple hidden factors ! :D 
    
    1. We use kronecker products to transform those multiple hidden dimensions into a single one (vec space)
    2. We make  computations with this single hidden dimension
    3. We flip back to the "source space" after the trial to update the B and D matrices !
    """    
    normA,normB,normD = vectorize_weights(inA,inB,inD,inU)
        # These weights are the same across the whole training
    
    def _scan_training(carry,key):
        key,trial_key = jr.split(key)
        
        pa,pb,pc,pd,pe = carry  # Dirichlet weights for this trial
        
        # From dirichlet weights to mdp parameters !
        trial_a,trial_b,trial_d = vectorize_weights(pa,pb,pd,inU)  # normalize & vectorize !
        trial_c,trial_e = to_log_space(pc,pe)
        
        trial_a_nov,trial_b_nov = get_vectorized_novelty(pa,pb,inU,compute_a_novelty=True,compute_b_novelty=True)
                    # get novelty scores, if needed
                    
        # T timesteps happen below : 
        [obs_darr,obs_arr,obs_vect_arr,
         true_s_darr,true_s_arr,true_s_vect_arr,
         u_d_arr,u_arr,u_vect_arr,
         qs_arr,qpi_arr,efes] = synthetic_trial_direct_run(trial_key,
                            Ns,Nos,Np,
                            trial_a,trial_a_nov,
                            trial_b,trial_b_nov,
                            trial_c,trial_d,trial_e,
                            normA,normB,normD,
                            T=T,Th =Th,
                            alpha = alpha,gamma = gamma, selection_method=selection_method,
                            planning_options=planning_options)
        
        a_post,b_post,c_post,d_post,e_post = learn_after_trial(obs_vect_arr,qs_arr,u_vect_arr,
                                                 pa,pb,pc,pd,pe,inU,
                                                 learn_what=learn_dictionnary["bool"],
                                                 learn_rates=learn_dictionnary["rates"],
                                                 post_trial_smooth=learn_dictionnary["smooth_states"])
        
        # a_post,b_post,d_post = pa,pb,pd
        return (a_post,b_post,c_post,d_post,e_post),(
                    obs_darr,obs_arr,obs_vect_arr,
                    true_s_darr,true_s_arr,true_s_vect_arr,
                    u_d_arr,u_arr,u_vect_arr,
                    qs_arr,qpi_arr,efes,
                    a_post,b_post,c_post,d_post,e_post)
        
    
    next_keys = jr.split(rngkey, Ntrials)
    (_,_,_,_,_), (
        all_obs_darr,all_obs_arr,all_obs_vect_arr,
        all_true_s_darr,all_true_s_arr,all_true_s_vect_arr,
        all_u_d_arr,all_u_arr,all_u_vect_arr,
        all_qs_arr,all_qpi_arr,efes_arr,
        a_hist,b_hist,c_hist,d_hist,e_hist) = jax.lax.scan(_scan_training, (a0,b0,c,d0,e),next_keys)
    
    return c_hist
    # return [all_obs_vect_arr,all_true_s_arr,all_u_vect_arr,all_qs_arr,all_qpi_arr,efes_arr,a_hist,b_hist,d_hist]
    return [all_obs_arr,all_true_s_arr,all_u_arr,all_qs_arr,all_qpi_arr,efes_arr,a_hist,b_hist,d_hist]

def learn_example():
    # Environment variables
    Nsubjects = 1
    Ntrials = 10
    T = 3
    Th = 3
    
    pHA = 0.8
    pWin = 0.99
    A,B,C,D,E,U = get_T_maze_gen_process(0.8,pHA,pWin)
    
    init_belief_pHA = 0.5
    a,d=get_T_maze_model(init_belief_pHA,pWin,100.0)
    # b = [0.1*np.ones_like(Bf)+Bf for Bf in B]
    b = [200*Bf for Bf in B]
    
    learn_dictionnary = get_learning_options(True,True,True,run_smoother=True)
    
    rngkey = jax.random.PRNGKey(26)
    Nos = [Ai.shape[0] for Ai in A]
    Np = U.shape[0]
    Ns = 8 # 4 x 2, should be automated
    source_space_shape = (4,2)   
    
    [all_obs_arr,all_true_s_arr,all_u_arr,all_qs_arr,all_qpi_arr,efes_arr,a_hist,b_hist,d_hist] = run_training_multi_factors(rngkey,Ns,Nos,Np,
            Ntrials,T,
            A,B,D,U,
            a,b,C,d,E,
            Th = Th,
            selection_method="stochastic",alpha = 16,gamma = None, 
            planning_options=DEFAULT_PLANNING_OPTIONS,
            learn_dictionnary=learn_dictionnary)
    
    
    trial_index = 0
    
    # To learn, we need : 
    # 1. an history of observations, vectorized : 
    obs_hist = [o_mod[trial_index] for o_mod in all_obs_arr]
    # print(obs_hist)
    
    # 2. an history of selected actions, vectorized : 
    u_hist = all_u_arr[trial_index]
    # print(u_hist)
    
    # 3. an history of state inferences, vectorized : 
    qs_hist = all_qs_arr[trial_index]
    
    # action_shapes = [B_f.shape[-1] for B_f in B]
    
    # Nf = len(B)



def predict_next(explored_state,predictive_prior,observation_mappping):
    # Compute the expected observations under this true state and how it will affect
    # the posterior based on our current predictive prior

    # First, we compute the predictive observation for this prior
    def compute_predictive_observation(_obs_mapping,_predictive_posterior):
        _qo = tree_map(lambda a_m : jnp.einsum('oi,i->o',a_m,_predictive_posterior),_obs_mapping)
        return _qo
    
    # Predictive observation based on state_prior realization
    po = compute_predictive_observation(observation_mappping,explored_state)
    
    # Use it to compute the expected posterior if that happens : 
    qs,F = compute_state_posterior(predictive_prior,po,observation_mappping)
    
    return qs,po,F
    
def EFE_branch(t,
                previous_posterior,action_performed,
                predictive_posterior,observations,
                vecA,vecB,vecC,
                vecA_nov,vecB_nov,
                option_a_nov=True,option_b_nov=False,
                additional_options_planning=False):
        
    Gt = compute_Gt_array(t,observations,predictive_posterior,previous_posterior,action_performed,
                          vecA,vecA_nov,vecB,vecB_nov,vecC,
                          option_a_nov,option_b_nov,additional_options_planning)
    return Gt

# Vectorize this along all tree nodes for a specific explored future tmstp
def compute_efe_node(t,
            previous_posterior,
            vecA,vecB,vecC,vecE,
            vecA_nov,vecB_nov,
            trial_end_scalar,   
            option_a_nov=True,option_b_nov=False,
            additional_options_planning=False
            ):
    # For 
    # state_branch is [Ns]
    # prior_branch is [Ns]
    # action_performed_branch is [Np]
    # previous_posterior is [Ns]
    # Remember, we might be at the end of the trial. If we are, trial_end_scalar = 0 and G(pi_t) = E 
    
    Np = vecB.shape[-1]
    
    # In this node, these are the allowable actions :
    allowable_actions = jnp.arange(Np)
        
    # For each of these allowable actions, there is a corresponding predictive prior :
    def get_EFE(_action_idx):
        _action_vector = jax.nn.one_hot(_action_idx,Np)
        
        # 1. What is the predicted state if we perform action _action_idx ?
        _predicted_posterior = jnp.einsum('iju,j,u->i',vecB,previous_posterior,_action_vector)
        
        # 2. This would result in the following observations !
        _predicted_observation = tree_map(lambda a_m : jnp.einsum('oi,i->o',a_m,_predicted_posterior),vecA)
        
        # 3. And thus the following EFE !
        _efe_action = compute_Gt_array(t,_predicted_observation,_predicted_posterior,previous_posterior,_action_vector,
                    vecA,vecA_nov,
                    vecB,vecB_nov,
                    vecC,
                    option_a_nov,option_b_nov,additional_options_planning)
        return _efe_action,_predicted_posterior
    
    EFE_pi_tplus,qs_pi_tplus = vmap(get_EFE)(allowable_actions)
    # EFE_pi_tplus is [Np]
    # qs_pi_tplus is [Np x Ns]

    return qs_pi_tplus,trial_end_scalar*jnp.sum(EFE_pi_tplus,axis=-1) + vecE







@partial(jit,static_argnames=["Ph","Sh","remainder_state_bool"])
def expand_tree(current_EFE_node,q_previous,
                vecA,vecB,
                Ph,Sh,
                remainder_state_bool=True):
    Ns = q_previous.shape[-1]
    Np = current_EFE_node.shape[-1]
    
    # 1.
    # Explore Ph different action paths based on their relative EFE
    # Outputs some predictive priors for the next timestep
    def pick_future_action_paths(_efe,_qs_t):
        # current EFE has been computed for this timestep, now, let's pick 
        # the most plausible ones !
        # current_posterior : [Ns]
        # current_EFE : [Np]
        # returns : 
        # next_priors [Np x Ns], policies_explored[Np]
                
        scanner = jnp.argsort(-_efe)
        
        _policies_explored = scanner[::-1][:Ph]
        _policies_explored = jnp.arange(Np)
        
        _mapped_policies = vmap(lambda x : jax.nn.one_hot(x,Np))(_policies_explored)
        
        _next_priors = vmap(lambda _p : jnp.einsum("iju,j,u->i",vecB,_qs_t,_p))(_mapped_policies)
        
        return _next_priors,_mapped_policies
    
    # 2.
    # Decompose each predictive priors into Sh individual states !
    # If remainder_state is set to True, also group the remaining unexplored states
    # together and compute its EFE too !
    
    def pick_future_state_paths(predictive_posterior):
        # predictive_posterior : [Ph]
        # action_explored : scalar
        
        # scanner = jnp.argsort(predictive_posterior)
        scanner = jnp.arange(Ns)
        
        mapped_states = (vmap(lambda x : jax.nn.one_hot(x,Ns)))(scanner)
        mapped_densities = (vmap(lambda x : predictive_posterior[x])(scanner))
        
        # mapped_states = (vmap(lambda x : jax.nn.one_hot(x,Ns))(scanner[::-1][:Sh]))
        # mapped_densities = (vmap(lambda x : predictive_posterior[x])(scanner[::-1][:Sh]))

        if (remainder_state_bool and (Sh < Ns)):
            # Get the remaining unexplored predictions :
            explored_filter = mapped_states.sum(axis=-2)
            unexplored_filter = jnp.ones_like(explored_filter) - explored_filter
            
            # This is gonna be ugly :D
            remainder_density = jnp.array([1.0 - jnp.sum(mapped_densities)])
            
            
            remainder_step1,_ = _normalize((predictive_posterior+1e-14)*unexplored_filter)
            # BUT :  We don't want any already explored state here !
            remainder_step2,_ = _normalize(remainder_step1*unexplored_filter)
            # Why ? If predictive_posterior*unexplored_filter is full 0, 
            # the normalization doesn't work as expected
            
            remainder = jnp.expand_dims(remainder_step2,-2)
            
            mapped_states = jnp.concatenate([mapped_states,remainder],axis=0)
            mapped_densities = jnp.concatenate([mapped_densities,remainder_density],axis=0)
        
        return mapped_states,mapped_densities  # Either Ns x (Sh) or Ns x (Sh + 1)
    
    # 3. (might be better outside this function)
    # + Compute the expected observations under this state and how it will affect
    # the posterior
    def next_posterior(_explored_state,_pred_post):
        # Predictive observation based on explored_state realization
        po = tree_map(lambda a_m : jnp.einsum('oi,i->o',a_m,_explored_state),vecA)
        
        # Use it to compute the expected posterior if that happens : 
        qs,F = compute_state_posterior(_pred_post,po,vecA)
        
        return qs
    
    # This is the same quantity computed in compute_node, but it may be smaller if Ph < Np
    next_priors,action_branches_explored = pick_future_action_paths(current_EFE_node,q_previous)
        #  p_next is of shape Ph x Ns !
        #  action_branches_explored is of shape Ph !
    
    # For each action, we take the Sh most likely states (and possibly the joint distribution of the remaining ones)
    next_potential_states,next_branches_densities = vmap(pick_future_state_paths)(next_priors)
        # next_potential_states is of shape Ph x (Sh(+1?)) x Ns !
        # This is used to compute observation predictions
        
    # We want to vectorize explore_one_path_function along : 
    # 1 dimension for the predictive posterior argument
    # 2 dimensions for the explored state argument 
    vectorized_next_posterior = vmap(vmap(next_posterior, in_axes=(0,None)))
    
    next_posteriors = vectorized_next_posterior(next_potential_states,next_priors)
        # prior_states is also of shape Ph x (Sh(+1?)) x Ns !
        # And this is used as a predictive posterior
        
    return next_branches_densities,next_priors,next_posteriors,action_branches_explored


def shape_trials(a,b):
    Ns = 5
    K = 2
    initial = jnp.array([
        [1,2,3,4,5],
        [6,7,8,9,10.0]
    ])
    
    
    
    # jnp.repeat(jnp.expand_dims(jnp.arange(Ns),0),K,0)
    print(initial.shape)
    
    def expansion(_init_x,adder):
        def expand_dim_a(x):
            return jnp.einsum("i,j->ji",x,jnp.ones(a))
        
        def expand_dim_b(x):
            return jnp.einsum("ij,k->kij",x,jnp.ones(b))
    
        transformed = vmap(expand_dim_a)(_init_x)
        transformed = vmap(expand_dim_b)(transformed)
        return transformed
        
    
    
    Th = 2
    for expansion_id in range(Th):
        val =  expansion(initial)
        final = jnp.reshape(val,(-1,Ns))
        initial = final
    print(final.shape)
    shapes = (a,b)*Th
    print(jnp.reshape(final,shapes+(K,Ns)))
    
   
def test(vecA,vecB):
    current_EFE_node = jnp.array([[-5.1248,-5.5297,-5.5297,-4.4317]])
    q_previous = jnp.array([0.5,0.5,0.0,0.0,0.0,0.0,0.0,0.0])
    Ph = 4
    Sh = 8
    Np = 4
    Ns = 8
    remainder_state_bool = False
    
    def pick_future_action_paths(_efe,_qs_t):
        # current EFE has been computed for this timestep, now, let's pick 
        # the most plausible ones !
        # current_posterior : [Ns]
        # current_EFE : [Np]
        # returns : 
        # next_priors [Np x Ns], policies_explored[Np]
                
        scanner = jnp.argsort(-_efe)
        
        _policies_explored = scanner[::-1][:Ph]
        _policies_explored = jnp.arange(Np)
        
        _mapped_policies = vmap(lambda x : jax.nn.one_hot(x,Np))(_policies_explored)
        
        _next_priors = vmap(lambda _p : jnp.einsum("iju,j,u->i",vecB,_qs_t,_p))(_mapped_policies)
        
        return _next_priors,_mapped_policies
    
    # 2.
    # Decompose each predictive priors into Sh individual states !
    # If remainder_state is set to True, also group the remaining unexplored states
    # together and compute its EFE too !
    
    def pick_future_state_paths(predictive_posterior):
        # predictive_posterior : [Ph]
        # action_explored : scalar
        
        # scanner = jnp.argsort(predictive_posterior)
        scanner = jnp.arange(Ns)
        mapped_states = (vmap(lambda x : jax.nn.one_hot(x,Ns)))(scanner)
        mapped_densities = (vmap(lambda x : predictive_posterior[x])(scanner))
        # mapped_states = (vmap(lambda x : jax.nn.one_hot(x,Ns))(scanner[::-1][:Sh]))
        # mapped_densities = (vmap(lambda x : predictive_posterior[x])(scanner[::-1][:Sh]))

        if (remainder_state_bool and (Sh < Ns)):
            # Get the remaining unexplored predictions :
            explored_filter = mapped_states.sum(axis=-2)
            unexplored_filter = jnp.ones_like(explored_filter) - explored_filter
            
            # This is gonna be ugly :D
            remainder_density = jnp.array([1.0 - jnp.sum(mapped_densities)])
            
            
            remainder_step1,_ = _normalize((predictive_posterior+1e-14)*unexplored_filter)
            # BUT :  We don't want any already explored state here !
            remainder_step2,_ = _normalize(remainder_step1*unexplored_filter)
            # Why ? If predictive_posterior*unexplored_filter is full 0, 
            # the normalization doesn't work as expected
            
            remainder = jnp.expand_dims(remainder_step2,-2)
            
            mapped_states = jnp.concatenate([mapped_states,remainder],axis=0)
            mapped_densities = jnp.concatenate([mapped_densities,remainder_density],axis=0)
        
        return mapped_states,mapped_densities  # Either Ns x (Sh) or Ns x (Sh + 1)
    
    # 3. (might be better outside this function)
    # + Compute the expected observations under this state and how it will affect
    # the posterior
    def next_posterior(_explored_state,_pred_post):
        # Predictive observation based on explored_state realization
        po = tree_map(lambda a_m : jnp.einsum('oi,i->o',a_m,_explored_state),vecA)
        
        # Use it to compute the expected posterior if that happens : 
        qs,F = compute_state_posterior(_pred_post,po,vecA)
        
        return qs,po,F
    
    # This is the same quantity computed in compute_node, but it may be smaller if Ph < Np
    next_priors,action_branches_explored = pick_future_action_paths(current_EFE_node,q_previous)
        #  p_next is of shape Ph x Ns !
        #  action_branches_explored is of shape Ph !
    
    print(next_priors,action_branches_explored)
    
    # For each action, we take the Sh most likely states (and possibly the joint distribution of the remaining ones)
    next_potential_states,next_branches_densities = vmap(pick_future_state_paths)(next_priors)
        # next_potential_states is of shape Ph x (Sh(+1?)) x Ns !
        # This is used to compute observation predictions
    
    print(next_potential_states)    
    print(next_branches_densities)
    # We want to vectorize explore_one_path_function along : 
    # 1 dimension for the predictive posterior argument
    # 2 dimensions for the explored state argument 
    vectorized_next_posterior = vmap(vmap(next_posterior, in_axes=(0,None)))
    
    
    
    
    next_posteriors,o,f = vectorized_next_posterior(next_potential_states,next_priors)
        # prior_states is also of shape Ph x (Sh(+1?)) x Ns !
        # And this is used as a predictive posterior
    for action in range(Np):
        for state in range(Ns):
            print("Action "+str(action+1)+" , state "+str(state+1)+"")
            print(np.round(np.array(next_priors[action]),2))
            print(np.round(np.array(next_potential_states[action,state]),2))
            print("###")
            print(o[0][action,state])
            print("-")
            print(o[1][action,state])
            print("=")
            print(np.round(np.array(next_posteriors[action,state]),2))
    
    
    
@partial(jit,static_argnames=["Th","Sh","Ph","option_a_nov",'option_b_nov','additional_options_planning','explore_remaining_paths'])
def this_works(qs_current,start_t,
        vecA,vecB,vecC,vecE,
        nov_a,nov_b,
        filter_trial_end,
        Th,Sh,Ph,
        option_a_nov=True,option_b_nov=False,
        additional_options_planning=False,explore_remaining_paths=False):
    EFE_FLOOR = -10000
    Ns = qs_current.shape[-1]
    Np = vecB.shape[-1]
    
    # Utils : 
    # These are the predicted shapes for the next timestep
    # We only use the computed EFE here !
    if Sh < Ns:
        if explore_remaining_paths:
            exploration_step_shape = (Ph,Sh+1)
        else : 
            exploration_step_shape = (Ph,Sh)
    else :
        exploration_step_shape = (Ph,Ns)
    
    
    
    # Forward tree building
    # 2 philosophies for this problem : 
    # 1. We want to use scan, and therefore need a constant shape for each prospective step
    #      thus we precompute an array of paths we want to explore, and then scan them
    # 2. We want to reduce redundant computation and therefore have an expanding tree. 
    #      it seems that using lists (or maybe pytrees ?) works best in that case

    exploration_tree = []  # We will fill this tree with Th branches
    
    # First tiemstep EFE computation
    qs_pi = qs_current
    
    # qs_pi is K x [Ns]
    compute_node_func = (lambda _qs : compute_efe_node(start_t,_qs,
                                            vecA,vecB,vecC,vecE,
                                            nov_a,nov_b,
                                            filter_trial_end[start_t],
                                            option_a_nov,option_b_nov,additional_options_planning))
    qs_next,efe_next_actions = vmap(compute_node_func)(qs_pi)
    exploration_tree = [[qs_pi,efe_next_actions,None,None]]
        # Predictive priors & efe for this timestep
        #      K x  [Np x Ns]  |  K x [Np]
    
    qs_previous = qs_pi  # This posterior will also be used for subsequent estimations
    
    # Building the recursive tree : 
    for explorative_timestep in range(1,Th+1): 
        t = start_t+explorative_timestep
                      
        # 1. expand the tree :
        expand_tree_func = (lambda _efe,_qprev : expand_tree(_efe,_qprev,
                                                    vecA,vecB,
                                                    Ph,Sh,
                                                    remainder_state_bool=explore_remaining_paths))
        state_branch_densities,new_priors,new_posteriors,new_ut = vmap(expand_tree_func)(efe_next_actions,qs_previous)       
        
        
        # 2. Flatten the tree branches : 
        # For the new branches, the posterior is new_posteriors K x [Ns x Np] x Ns
        #                                   (previous branches) - (new branches) - Dist
        qs_pi = jnp.reshape(new_posteriors,(-1,Ns))
        
        # 3. Compute the efe for this new branch !
        compute_node_func = (lambda _qs : compute_efe_node(t,_qs,
                                            vecA,vecB,vecC,vecE,
                                            nov_a,nov_b,
                                            filter_trial_end[t],
                                            option_a_nov,option_b_nov,additional_options_planning))
        qs_next,efe_next_actions = vmap(compute_node_func)(qs_pi)
        
        exploration_tree.append([qs_pi,efe_next_actions,state_branch_densities,new_ut])
        qs_previous = qs_pi
        
        # # debug
        # if t==0:
        #     reformated_efes = jnp.reshape(efe_next_actions,(Np,Sh) + (Np,))
        #     reformated_priors = jnp.reshape(new_priors,(Np,) + (Ns,))
        #     reformated_posterior = jnp.reshape(qs_pi,(Np,Ns) + (Ns,))
        #     for action in range(Np):
        #         for state in range(Ns):
        #             print("Action "+str(action+1)+" , state "+str(state+1)+"-----------------------------")
        #             print("EFE = " + str(np.round(np.array(reformated_efes[action,state,:]),2)))
        #             print("Prior = " + str(np.round(np.array(reformated_priors[action,:]),2)))
        #             print("Posterior = " + str(np.round(np.array(reformated_posterior[action,state,:]),2)))
     
    # space_tuple_next = (exploration_step_shape*(Th-1))
    # [_,efe_next_tsmtp,_,_] = exploration_tree[-1]
    # efe_next_tsmtp = jnp.reshape(efe_next_tsmtp,space_tuple_next+(Np,))
    
    
    def remerge_action(_efe_children,_children_ut):
        # We use children_ut as a mapping tool from efe_children back to efe_parents: 
        carry_efe = jnp.einsum("abc,ab->ac",_children_ut,_efe_children)  
                # Efe of subsequent next steps for the explored action paths
        carry_efe = jnp.where(carry_efe==0,EFE_FLOOR,carry_efe)
        
        return carry_efe
    
    carry_efe = jnp.zeros_like(exploration_tree[-1][1])
    for explorative_timestep in range(Th,0,-1): 
        space_tuple_next = (exploration_step_shape*(explorative_timestep))

        # These are the predicted values for the next timestep
        # We only use the computed EFE here !
        [qs_tsmtp,efe_this_tsmtp,state_branch_densities,ut_next_tsmtp] = exploration_tree[explorative_timestep]
        state_branch_densities = jnp.reshape(state_branch_densities,space_tuple_next)
        efe_this_tsmtp = jnp.reshape(efe_this_tsmtp + carry_efe,space_tuple_next+(Np,))
                
        # We marginalize the efe for the next timestep across expected actions ... 
        margin_efe = jnp.sum(efe_this_tsmtp*jax.nn.softmax(efe_this_tsmtp,axis=-1),axis=-1)
        # ... and states
        margin_efe_next_tmstp = jnp.sum(state_branch_densities*margin_efe,axis=-1)

        # To get a quantity that can be added to the (previous) explorative timestep, it
        # has to map to the policy axis. To do this, we use our history of the explored action !
        # The unexplored actions should have EFE = -inf.
        flattened_margin_efe = jnp.reshape(margin_efe_next_tmstp,(-1,Ph))
        
        # next_timestep_shape = (exploration_step_shape*(explorative_timestep-1)) + (Np,)
        carry_efe = remerge_action(flattened_margin_efe,ut_next_tsmtp)

    
    final_efe = carry_efe+exploration_tree[0][1]
    
    return final_efe
    
    
if __name__ == '__main__': 
    
    
    # Model of the task : 
    pContext = 0.0
    pHA = 1.0
    pWin = 0.98
    A,B,D,E,U = get_T_maze_gen_process(pContext,pHA,pWin)
    
    init_belief_pHA = 1.0
    initial_hint_confidence = 200
    la = -4
    rs = 2
    a,b,c,d =get_T_maze_model(init_belief_pHA,pWin,initial_hint_confidence,la,rs)
    
    learn_dictionnary = get_learning_options(True,False,True,run_smoother=True)
    
    rngkey = jax.random.PRNGKey(206)
    Nos = [Ai.shape[0] for Ai in A]
    Np = U.shape[0]
    Ns = 8 # 4 x 2, should be automated
    source_space_shape = (4,2)   
    
    vecA,vecB,vecD = vectorize_weights(a,b,d,U)
    nov_a,nov_b = get_vectorized_novelty(a,b,U,compute_a_novelty=True,compute_b_novelty=True)
    trial_c,trial_e = to_log_space(c,E)
    
    
    
    # print(vecA)
    # print(vecB[...,0])
    # exit()
    # test(vecA,vecB)
    # exit()
    
    print("___________________________________________________________________")
    qs_current = jnp.array([[0.5,0.5,0,0,0,0.0,0.0,0.0]])
    start_t = 0
    vecC = trial_c
    vecE = trial_e
    filter_trial_end = jnp.array([1.0,1.0,0.0])
    Th = 2
    Sh = 8
    Ph = 4
    
    option_a_nov = False
    option_b_nov = False
    additional_options_planning = False    
    remainder_state_bool = True
    
    
    g = this_works(qs_current,start_t,
        vecA,vecB,vecC,vecE,
        nov_a,nov_b,
        filter_trial_end,
        Th,Sh,Ph,
        option_a_nov=option_a_nov,option_b_nov=option_b_nov,
        additional_options_planning=additional_options_planning,explore_remaining_paths=remainder_state_bool)
    print(g)
    
    
    partialled_this_works = partial(this_works,start_t=start_t,
        vecA=vecA,vecB=vecB,vecC=vecC,vecE=vecE,
        nov_a=nov_a,nov_b=nov_b,
        filter_trial_end=filter_trial_end,
        Th=Th,Sh=Sh,Ph=Ph,
        option_a_nov=option_a_nov,option_b_nov=option_b_nov,
        additional_options_planning=additional_options_planning,explore_remaining_paths=remainder_state_bool)
    
    # print(partialled_this_works(qs_current))
    
    N = 300
    a_lot_of_random_vectors,_ = _normalize(jax.random.normal(rngkey,(N,1,8)))
    # print(a_lot_of_random_vectors)
    print(vmap(partialled_this_works)(a_lot_of_random_vectors))
    
    # print(jax.grad(partialled_this_works)(qs_current))
    exit()
    # Environment variables
    Nsubjects = 1
    Ntrials = 1
    T = 3
    Th = 2 # Temporal horizon
    Ph = 4 # Policy horizon
    Sh = 8 # State horizon
    # The exploration tree will have size ~ (Ph x (Sh(+1?)))**Th
    
    # To avoid recompiling for each t :
    filter_trial_end = jnp.ones((Th,))
    filter_trial_end = jnp.array([1.0,1.0,0.0])
    
    # Forward tree building
    # 2 philosophies for this problem : 
    # 1. We want to use scan, and therefore need a constant shape for each prospective step
    #      thus we precompute an array of paths we want to explore, and then scan them
    # 2. We want to reduce redundant computation and therefore have an expanding tree. 
    #      it seems that using lists (or maybe pytrees ?) works best in that case
    option_a_nov = False
    option_b_nov = False
    additional_options_planning = False    
    remainder_state_bool = True

    exploration_tree = []  # We will fill this tree with Th branches
    
    start_t = 0
    
    
    # First inversion and EFE computation
    t = start_t
    qs_current = jnp.array([[0.5,0.5,0,0,0,0.0,0.0,0.0]])
    
    
    qs_pi = qs_current
    
    
    # qs_previous is K x [Ns]
    compute_node_func = (lambda _qs : compute_efe_node(t,_qs,
                                            vecA,vecB,trial_c,trial_e,
                                            nov_a,nov_b,
                                            filter_trial_end[t],
                                            option_a_nov,option_b_nov,additional_options_planning))
    qs_next,efe_next_actions = vmap(compute_node_func)(qs_pi)
        # Predictive priors & efe for this timestep
        #      K x  [Np x Ns]  |  K x [Np]
    qs_previous = qs_pi
    
    exploration_tree = [[qs_pi,efe_next_actions,None,None]]
    # 0. Flatten the previous branches : 
    # qs_previous = jnp.reshape(new_posteriors,(-1,Ns))
    

    # Building the recursive tree : 
    for explorative_timestep in range(1,Th+1): 
        t = start_t+explorative_timestep
                      
        # 1. expand the tree :
        expand_tree_func = (lambda _efe,_qprev : expand_tree(_efe,_qprev,
                                                    vecA,vecB,
                                                    Ph,Sh,
                                                    remainder_state_bool=remainder_state_bool))
        state_branch_densities,new_priors,new_posteriors,new_ut = vmap(expand_tree_func)(efe_next_actions,qs_previous)       
        
        
        # 2. Flatten the tree branches : 
        # For the new branches, the posterior is new_posteriors K x [Ns x Np] x Ns
        #                                   (previous branches) - (new branches) - Dist
        qs_pi = jnp.reshape(new_posteriors,(-1,Ns))
        
        # 3. Compute the efe for this new branch !
        compute_node_func = (lambda _qs : compute_efe_node(t,_qs,
                                            vecA,vecB,trial_c,trial_e,
                                            nov_a,nov_b,
                                            filter_trial_end[t],
                                            option_a_nov,option_b_nov,additional_options_planning))
        qs_next,efe_next_actions = vmap(compute_node_func)(qs_pi)
        
        exploration_tree.append([qs_pi,efe_next_actions,state_branch_densities,new_ut])
        qs_previous = qs_pi
        
        # # debug
        # if t==0:
        #     reformated_efes = jnp.reshape(efe_next_actions,(Np,Sh) + (Np,))
        #     reformated_priors = jnp.reshape(new_priors,(Np,) + (Ns,))
        #     reformated_posterior = jnp.reshape(qs_pi,(Np,Ns) + (Ns,))
        #     for action in range(Np):
        #         for state in range(Ns):
        #             print("Action "+str(action+1)+" , state "+str(state+1)+"-----------------------------")
        #             print("EFE = " + str(np.round(np.array(reformated_efes[action,state,:]),2)))
        #             print("Prior = " + str(np.round(np.array(reformated_priors[action,:]),2)))
        #             print("Posterior = " + str(np.round(np.array(reformated_posterior[action,state,:]),2)))
                    
        # if t==1:
        #     reformated_efes = jnp.reshape(efe_next_actions,(Np,Sh) + (Np,))
        #     reformated_qs_previous = jnp.reshape(qs_previous,(Np,Sh) + (Ns,))
        #     print("--------------------------------")
        #     print(reformated_qs_previous.shape)
        #     for action in range(Np):
        #         for state in range(Ns):
        #             print("Action "+str(action+1)+" , state "+str(state+1)+"")
        #             print(np.round(np.array(reformated_efes[action,state,:]),2))
        #             print(np.round(np.array(reformated_qs_previous[action,state,:]),2))
            # # print(np.round(np.array(qs_next),2))
            # for action in range(Np):
            #     for state in range(Ns):
            #     print(efe_next_actions)
            # exit()
            
        

        # if t==0:
        #     reformated_posteriors = jnp.reshape(new_posteriors,(Np,Sh) + (Ns,))
        #     for action in range(Np):
        #         for state in range(Ns):
        #             print("Action "+str(action+1)+" , state "+str(state+1)+"")
        #             print(np.round(np.array(reformated_posteriors[action,state]),2))
        #     # exit()
    # exit()
        
    # These are the predicted shapes for the next timestep
    # We only use the computed EFE here !
    if Sh < Ns:
        if remainder_state_bool:
            exploration_step_shape = (Ph,Sh+1)
        else : 
            exploration_step_shape = (Ph,Sh)
    else :
        exploration_step_shape = (Ph,Ns)
     
    # space_tuple_next = (exploration_step_shape*(Th-1))
    # [_,efe_next_tsmtp,_,_] = exploration_tree[-1]
    # efe_next_tsmtp = jnp.reshape(efe_next_tsmtp,space_tuple_next+(Np,))
    
    
    def remerge_action(efe_children,children_ut,BOTTOM = -10000):
        # We use children_ut as a mapping tool from efe_children back to efe_parents: 
        carry_efe = jnp.einsum("abc,ab->ac",children_ut,efe_children)  
                # Efe of subsequent next steps for the explored action paths
        carry_efe = jnp.where(carry_efe==0,BOTTOM,carry_efe)
        
        return carry_efe
    
    carry_efe = jnp.zeros_like(exploration_tree[-1][1])
    for explorative_timestep in range(Th,0,-1): 
        space_tuple_next = (exploration_step_shape*(explorative_timestep))

        # These are the predicted values for the next timestep
        # We only use the computed EFE here !
        [qs_tsmtp,efe_this_tsmtp,state_branch_densities,ut_next_tsmtp] = exploration_tree[explorative_timestep]
        state_branch_densities = jnp.reshape(state_branch_densities,space_tuple_next)
        efe_this_tsmtp = jnp.reshape(efe_this_tsmtp + carry_efe,space_tuple_next+(Np,))
                
        # We marginalize the efe for the next timestep across expected actions ... 
        margin_efe = jnp.sum(efe_this_tsmtp*jax.nn.softmax(efe_this_tsmtp,axis=-1),axis=-1)
        # ... and states
        margin_efe_next_tmstp = jnp.sum(state_branch_densities*margin_efe,axis=-1)

        # To get a quantity that can be added to the (previous) explorative timestep, it
        # has to map to the policy axis. To do this, we use our history of the explored action !
        # The unexplored actions should have EFE = -inf.
        flattened_margin_efe = jnp.reshape(margin_efe_next_tmstp,(-1,Ph))
        
        # next_timestep_shape = (exploration_step_shape*(explorative_timestep-1)) + (Np,)
        carry_efe = remerge_action(flattened_margin_efe,ut_next_tsmtp)

    
    final_efe = carry_efe+exploration_tree[0][1]
    u = jax.nn.softmax(final_efe,axis=-1)
    print(u)
    # print(exploration_tree)
    exit()    
    space_tuple = ((Ph,Sh+1)*Th)
    
    out_ut = jnp.reshape(ut,space_tuple)
    print(out_ut[:,0,:,0,:,0])
    # print(jnp.reshape(ut,))
        
    exit()
    diagnostic = bruteforce_treesearch(0,
        q_init,
        vecA,vecB,trial_c,trial_e,
        nov_a,nov_b,
        filter_trial_end,
        Np,Th,Sh,
        efe_a_nov=False,efe_b_nov=False,efe_old_a_nov=False)
    
    print(diagnostic)
    exit()
    [all_obs_arr,all_true_s_arr,all_u_arr,all_qs_arr,all_qpi_arr,efes_arr,a_hist,b_hist,d_hist] = run_training_multi_factors(rngkey,Ns,Nos,Np,
            Ntrials,T,
            A,B,D,U,
            a,b,c,d,E,
            Th = Th,
            selection_method="stochastic",alpha = 16,gamma = None, 
            planning_options=DEFAULT_PLANNING_OPTIONS,
            learn_dictionnary=learn_dictionnary) 
    

    
    print(d_hist)
    print(np.round(np.array(all_qpi_arr),2))
    # for mod in range(2):
    #     print(np.round(np.array(a_hist[0][...,mod]),2))
    
    exit()
    from actynf.jax_methods.layer_learn import forwards_pass,backwards_pass,smooth_posterior_after_trial
    
    trial_idx = 0
    o = [o_m[trial_idx] for o_m in all_obs_arr]
    qs = all_qs_arr[trial_idx]
    act_h = all_u_arr[trial_idx]
    print()
    print(o)
    print(qs)
    print(act_h)
    
    print("-----------------------------")
    # ll_for,forwards_smooths = forwards_pass(
    #     o,qs,act_h,
    #     a,b,d,U
    # )
    # print(np.round(np.array(forwards_smooths),2))
    
    
    # print("----------------------------")
    # ll_back,backwards_smooths = backwards_pass(
    #     o,qs,act_h,
    #     a,b,d,U
    # )
    
    val = smooth_posterior_after_trial(o,qs,act_h,
        a,b,d,U,
        filter_type="two_filter")
    print(np.round(np.array(val),2))
    
    val = smooth_posterior_after_trial(o,qs,act_h,
        a,b,d,U,
        filter_type="one_filter")
    print(np.round(np.array(val),2))
    
    # print(np.round(np.array(backwards_smooths),2))
    # print("################################")
    # integrator = (lambda X : X/jnp.sum(X,axis=-1,keepdims=True))
    # smooth_posterior = integrator(forwards_smooths*backwards_smooths)
    # print(np.round(np.array(smooth_posterior),2))
    # print(ll_for)
    # print(ll_back)
    # print(smoothed_posteriors)
    # print(smoothed_posteriors.shape)
    
    # print(np.round(np.array(smoothed_posteriors),2))
    
    
    # ex_arr = jnp.array([
    #     [5,6],
    #     [9,7]
    # ])
    # print(forwards_smooths*backwards_smooths)
    # val,_ = _normalize(forwards_smooths*backwards_smooths,axis=-1)
    # print(np.round(np.array(val),2))