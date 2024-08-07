import jax.numpy as jnp
import jax.random as jr
import jax

import numpyro.distributions as distr

from jax.tree_util import tree_map
from jax import vmap
from jax import jit

from functools import partial
from itertools import product

from .jax_toolbox import _normalize,_jaxlog
from .planning_tools import compute_Gt_array,compute_novelty
from .layer_infer_state import compute_state_posterior

from .jax_gumbel import gumbel_softmax

# A set of functions for agents to plan their next moves.  Note that we use a bruteforce treesearch approach, which is
# a computational nightmare, but may be adapted to fit short term path planning.

@partial(jit,static_argnames=["option_a_nov","option_b_nov",'additional_options_planning'])
def compute_efe_one_action(action_posterior,previous_state_posterior,
            t,trial_end_scalar,
            vecA,vecB,vecC,vecE,
            vecA_nov,vecB_nov,   
            option_a_nov=True,option_b_nov=False,
            additional_options_planning=False):
    # 1. What is the expected distribution given the chosen policy qpi and perceived current state previous_posterior ?    
    # predicted_distribution is the expected state at the next tmstp given qpi(t)
    predicted_distribution = jnp.einsum("iju,j,u->i",vecB,previous_state_posterior,action_posterior)
        # Branched priors : [Ns] 
        # this should be used to compute the EFE !
    
    # 2. This predicted distribution would result in the following observations !
    _predicted_observation = tree_map(lambda a_m : jnp.einsum('oi,i->o',a_m,predicted_distribution),vecA)
    
    # 2. And thus the following EFE !
    _efe_action = compute_Gt_array(t,_predicted_observation,predicted_distribution,previous_state_posterior,action_posterior,
                vecA,vecA_nov,
                vecB,vecB_nov,
                vecC,
                option_a_nov,option_b_nov,additional_options_planning)
    
    action_log_posterior = jnp.sum(_efe_action)
    
    action_log_prior = jnp.sum(vecE*action_posterior) # Habits
    
    # Remember, we might be at the end of the trial. If we are, trial_end_scalar = 0 and G(pi_t) = E     
    G = action_log_prior + trial_end_scalar*action_log_posterior
    
    return  predicted_distribution, G
    
@partial(jit,static_argnames=["option_a_nov","option_b_nov",'additional_options_planning'])
def compute_efe_each_action(previous_posterior,
            t,trial_end_scalar,
            vecA,vecB,vecC,vecE,
            vecA_nov,vecB_nov,   
            option_a_nov=True,option_b_nov=False,
            additional_options_planning=False):
    # For :
    # state_branch is [Ns]
    # prior_branch is [Ns]
    # action_performed_branch is [Np]
    # previous_posterior is [Ns]
    # Remember, we might be at the end of the trial. If we are, trial_end_scalar = 0 and G(pi_t) = E 
    Np = vecB.shape[-1]
    
    # Constant : the action matrix explored (we compute the EFE of all actions)
    all_actions_matrix = jax.nn.one_hot(jnp.arange(Np),Np,axis=-2)
    
    efe_single_action_func = (lambda qpi : compute_efe_one_action(qpi,previous_posterior,
            t,trial_end_scalar,
            vecA,vecB,vecC,vecE,
            vecA_nov,vecB_nov,   
            option_a_nov=option_a_nov,option_b_nov=option_b_nov,
            additional_options_planning=additional_options_planning))
    
    predicted_distribution, G = vmap(efe_single_action_func)(all_actions_matrix)
    return  predicted_distribution, G

# _________________________________________________________________________________________
# Functions used to build the policy tree. 
#  --> Add a level to the tree (changing a tensor of shape K to shape K x (Ph x Sh))
#  --> Evaluate the EFE at a specific level of the tree

# "soft" branch selection functions
def branching_out_actions_gumbel(efe_each_action,N_action_branches,rngkey,
                         _action_sample_temperature = 0.001):
    # See papers like : Differentiable Subset Pruning of Transformer Heads, Jiaoda Li, Ryan Cotterell, Mrinmaya Sachan
    # Or implementations like : https://gist.github.com/rahular/6091da25c8c8ce32f6310ec7399a135b
        
    Np = efe_each_action.shape[-1]
    
    # We use a Gumbel-Softmax trick here to avoid 
    # split actions in case of similar EFEs !
    
    # A gumbel softmax powered top k operator :
    def action_brancher(carry,xs):
        remaining_logits = carry   # The EFEs yet to explore ! 
        rngkey_k = xs 
        
        # Softmax approximation of the sampled distribution :
        sampled_branch = gumbel_softmax(remaining_logits,_action_sample_temperature,rngkey_k)
       
        # Recursively penalize already explored branches :
        remaining_logits += _jaxlog(1.0-sampled_branch)  # from 1.0 (unexplored) to 0.0 (fully explored)
        
        return remaining_logits,(sampled_branch,remaining_logits)
    
    # Distribution over what actions we may explore, to be further branched !
    init_distribution_scan = efe_each_action
    
    _,(explored_actions,efes_after_sampling) = jax.lax.scan(action_brancher,init_distribution_scan,jr.split(rngkey,N_action_branches))
    
    # No remainder computations here : the unexplored actions are assumed to have very low (neg)EFE.
    return explored_actions,efes_after_sampling

def branching_out_states_gumbel(_pred_prior,N_state_branches,rngkey,
                         _state_remainder_branch=True,
                         _state_sample_temperature = 0.01):
    # See papers like : Differentiable Subset Pruning of Transformer Heads, Jiaoda Li, Ryan Cotterell, Mrinmaya Sachan
    # Or implementations like : https://gist.github.com/rahular/6091da25c8c8ce32f6310ec7399a135b
        
    EPSILON = 1e-10
    Ns, = _pred_prior.shape
    
    # We use a Gumbel-Softmax trick here to avoid 
    # split actions in case of similar EFEs !
    
    # A gumbel softmax powered top k operator :
    def state_brancher(carry,xs):
        remaining_logits,remaining_density = carry   # The logprobs yet to explore ! 
        rngkey_k = xs 
        
        # Softmax approximation of the sampled distribution :
        sampled_branch = gumbel_softmax(remaining_logits,_state_sample_temperature,rngkey_k)
        
        # Recursively penalize already explored branches :
        remaining_logits += _jaxlog(1.0-sampled_branch)   # from 1.0 (unexplored) to 0.0 (fully explored)
        
        
        # Compute the density of the explored path : 
        sampled_density = (remaining_density*sampled_branch)  
                    # The "quantity" of the possible future states explored 
        remaining_density = remaining_density - sampled_density
                    # This is the part of the distribution left unexplored
        sampled_density_sum = sampled_density.sum()
        
        
        return (remaining_logits,remaining_density),(sampled_branch,sampled_density_sum,remaining_logits)
    
    density_tracker = _pred_prior
    
    
    # Distribution over what actions we may explore, to be further branched !
    init_distribution_scan = (_jaxlog(_pred_prior),density_tracker)
    remaining_distribution,(explored_states,explored_densities,efes_after_sampling) = jax.lax.scan(state_brancher,init_distribution_scan,jr.split(rngkey,N_state_branches))
        
    # return remaining_distribution,unexplored_distribution
    if _state_remainder_branch:
        unexplored_mask = 1.0-explored_states.sum(axis=-2)
        unexplored_distribution = unexplored_mask*_pred_prior
        
        norm_remainder,_ = _normalize(unexplored_distribution+EPSILON)
        remainder_density = jnp.array([unexplored_distribution.sum()])
        
        last_explored_state = jnp.expand_dims(norm_remainder,axis=-2)
        
        

        # Add this to the previously explored paths
        explored_states = jnp.concatenate([explored_states,last_explored_state],axis=-2)
        explored_densities = jnp.concatenate([explored_densities,remainder_density],axis=-1)
    
    explored_densities,_ = _normalize(explored_densities) # Explored densities should sum to 1
    
    return explored_states,explored_densities
    # return explored_actions,efes_after_sampling

def branch_predictive_posterior(_predictive_prior,_branch_state_outcome,vecA):
    # Compute the expected observations under this state outcome
    # and how it will affect our posterior beliefs
    
    # Predictive observation based on explored_state realization
    po = tree_map(lambda a_m : jnp.einsum('oi,i->o',a_m,_branch_state_outcome),vecA)
    
    # Use it to compute the expected posterior if that happens : 
    qs,F = compute_state_posterior(_predictive_prior,po,vecA)
    
    return qs

# Tree building based on branch functions
def expand_branch(rngkey,G_branch,q_previous_branch,
                vecA,vecB,
                N_action_branches,_action_sample_temperature,
                N_state_branches,_state_sample_temperature,_state_remainder_branch):    
    # 1. Let's select which actions to explore :
    rngkey,rngkey_action = jr.split(rngkey)
    explored_actions,efes_after_sampling = branching_out_actions_gumbel(G_branch,N_action_branches,rngkey_action,
                                                            _action_sample_temperature = _action_sample_temperature)
        #  p_next is of shape Ph x Ns !
        #  action_branches_explored is of shape Ph !
    
    # 2. What are the corresponding next predictive priors ?
    pred_priors = vmap(lambda action : jnp.einsum("iju,j,u->i",vecB,q_previous_branch,action))(explored_actions)
    
    
    # 3. Let's explore individual realizations of these priors
    rngkey,rngkey_state = jr.split(rngkey)
    next_priors,next_densities = vmap(lambda prior : branching_out_states_gumbel(prior,N_state_branches,rngkey_state,
                                                            _state_remainder_branch=_state_remainder_branch,
                                                            _state_sample_temperature = _state_sample_temperature))(pred_priors)
        #  p_next is of shape Ph x Ns !
        #  action_branches_explored is of shape Ph !
    # reshaped_explored_actions = jnp.repeat(jnp.expand_dims(explored_actions,-3),next_priors.shape[-3],axis=-3)
    reshaped_explored_actions = explored_actions
    
    # 4. Compute the expected posteriors for each of those realizations ! 
    next_posteriors = vmap(vmap(lambda __pred_prior,__outcome : branch_predictive_posterior(__pred_prior,__outcome,vecA), in_axes=(None,0)))(pred_priors,next_priors)
    
    return next_posteriors,next_densities,reshaped_explored_actions

def expand_tree(rngkey,G_tree,q_previous_tree,
                vecA,vecB,
                N_action_branches,_action_sample_temperature,
                N_state_branches,_state_sample_temperature,_state_remainder_branch):
    Ns = q_previous_tree.shape[-1]
    Kt,Np = G_tree.shape  
      
    expand_branch_func = partial(expand_branch,
                vecA=vecA,vecB=vecB,
                N_action_branches=N_action_branches,_action_sample_temperature=_action_sample_temperature,
                N_state_branches=N_state_branches,_state_sample_temperature=_state_sample_temperature,_state_remainder_branch=_state_remainder_branch)
    rng_keys_all_branches = jr.split(rngkey,Kt)
    
    all_posteriors,all_densities,all_actions = vmap(expand_branch_func)(rng_keys_all_branches,G_tree,q_previous_tree)

    # Reshape so that these are 1 or 2 dimensionnal vectors :
    all_posteriors = jnp.reshape(all_posteriors,(-1,Ns))
    all_densities = jnp.reshape(all_densities,(-1,))
    # all_actions = jnp.reshape(all_actions,(-1,Np)) # Actions should stay 3dimensionnal
    return all_posteriors,all_densities,all_actions


@partial(jit,static_argnames=["Th","N_action_branches","N_state_branches",'_state_remainder_branch',"option_a_nov",'option_b_nov','additional_options_planning'])
def compute_EFE(rngkey,qs_current,start_t,
        vecA,vecB,vecC,vecE,
        nov_a,nov_b,
        filter_trial_end,
        Th,
        N_action_branches,_action_sample_temperature,
        N_state_branches,_state_sample_temperature,_state_remainder_branch,
        option_a_nov=True,option_b_nov=False,
        additional_options_planning=False):
    EFE_FLOOR = -10000
    
    Ns = qs_current.shape[-1]
    Np = vecB.shape[-1]
    
    # Forward tree building ----------------------------------------
    # 2 philosophies for this problem : 
    # 1. We want to use scan, and therefore need a constant shape for each prospective step
    #      thus we precompute an array of paths we want to explore, and then scan them (see layer_plan_classic.py)
    # 2. We want to reduce redundant computation and therefore have an expanding tree. 
    #      it seems that using lists (or maybe pytrees ?) works best in that case
    # 
    # This function uses an application of 2.
    
    
    # Functions used : 
    expand_tree_func = partial(expand_tree,vecA=vecA,vecB=vecB,
                N_action_branches=N_action_branches,_action_sample_temperature=_action_sample_temperature,
                N_state_branches=N_state_branches,_state_sample_temperature=_state_sample_temperature,_state_remainder_branch=_state_remainder_branch)
        # Function of rngkey,G_tree,q_previous_tree
    
    compute_node_efe_func = vmap(partial(compute_efe_each_action,
                vecA=vecA,vecB=vecB,vecC=vecC,vecE=vecE,
                vecA_nov=nov_a,vecB_nov=nov_b,   
                option_a_nov=option_a_nov,option_b_nov=option_b_nov,
                additional_options_planning=additional_options_planning),in_axes=(0,None,None))
        # Function of previous_posterior,t,trial_end_scalar
    



    # First timestep EFE computation
    qs_pi = qs_current
    qs_next_initial,G_tree = compute_node_efe_func(qs_current,start_t,filter_trial_end[0])  # Only one branch !

    exploration_tree = [{
            "posteriors" : qs_pi,
            "efe" : G_tree,
            "densities" : None,
            "actions" : None}]

    
    
    # The following steps will be unrolled by the compiler ! 
    # We expand the tree repeateadly until we reach the desired temporal horizon
    # Autobots, roll out !
    qs_previous = qs_pi  # This posterior will also be used for subsequent estimations
    for explorative_timestep in range(1,Th+1): 
        t = start_t+explorative_timestep # t in [start_t+1, current_t+Th]
                      
        # 1. expand the tree :
        rngkey,rngkey_expand_t = jr.split(rngkey)
        explored_posteriors,explored_densities,explored_actions = expand_tree_func(rngkey_expand_t,G_tree,qs_previous)
        
        # 2. Compute the efe for this new branch !
        _,G_tree = compute_node_efe_func(explored_posteriors,t,filter_trial_end[explorative_timestep])
        
        # G_tree = G_tree*filter_trial_end[explorative_timestep-1]  
        #         # If the previous timestep is trial end or after, this computation is not taken into account 
        #         # This is redundant, can we just remove it ?
        
        exploration_tree.append({
            "posteriors" : explored_posteriors,
            "efe" : G_tree,
            "densities" : explored_densities,
            "actions" : explored_actions})
        
        qs_previous = explored_posteriors
        
        
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
    
    
    # Backward tree summing ----------------------------------------
    def remerge_action(_efe_children,_children_ut,_eps=1e-10):
        """ 
        Map a K x Ph tensor of EFE onto a K x Np space using a K x Ph x Np mapping rule.
        Unexplored action paths should have a very low EFE !
        Difference with argsort based tree pruning : here, the same branch can be selected twice !
        
        """ 
        
        # Project _efes to the Full action space ,normalized by the total density
        # of the branch selected: 
        norm_projected_efe = jnp.einsum("abc,ab->ac",_children_ut,_efe_children)/jnp.sum(_children_ut+_eps,axis=-2)
        
        
        # A value of 0.0 to 1.0 rating how explored a specific action branch is overall:
        explored_filter = jnp.clip(jnp.sum(_children_ut+_eps,axis=-2),max=1.0)
        unexplored_filter = 1.0 - explored_filter
        unexplored_efe = unexplored_filter * EFE_FLOOR
        
        return norm_projected_efe + unexplored_efe
    
    
    # This will be unrolled ! (needs to be done sequentially, 
    # big Th values are obviously discouraged)
    # Autobots, roll out !       
    carry_efe = jnp.zeros_like(exploration_tree[Th]["efe"])
    # print(carry_efe.shape)
    for explorative_timestep in range(Th,0,-1): # Th -> Th-1 -> ... -> 1
        tree_t = exploration_tree[explorative_timestep]

        efe_this_tsmtp = tree_t["efe"] + carry_efe
                # Shape : K' x Np
        
        
        # Now, let's sum up this branch in order to transmit it to the parent branches !
        # 1st, we marginalize the efe for the next timestep across expected actions ... 
        # (should there be a precison parameter here ?)
        margin_efe_actions = jnp.sum(efe_this_tsmtp*jax.nn.softmax(efe_this_tsmtp,axis=-1),axis=-1)
                # margin_efe_actions is the expected efe for the subsequent timestep for each state branch
                # Shape : (K'/Sh = K*Ph)
        
        # ... and states
        margin_efe_actions_states = jnp.reshape(tree_t["densities"]*margin_efe_actions,(-1,N_state_branches)).sum(axis=-1)
                # Shape : K'/(Sh*Ph) = K
                
        # Reshape to explicitely show the various action branches : 
        parent_branches_efe = jnp.reshape(margin_efe_actions_states,(-1,N_action_branches))
                # Shape : K'/(Sh*Ph) = K
        # Include the unexplored actions ! 
        carry_efe = remerge_action(parent_branches_efe,tree_t["actions"])
    
    
    total_efe = carry_efe + exploration_tree[0]["efe"]
    
    # predictive posterior over the very next hidden state given this posterior : 
    u_post = jax.nn.softmax(total_efe,axis=-1)
    state_predictive_posterior = jnp.einsum("aus,au->s",qs_next_initial,u_post)
    return total_efe[0,...],u_post[0,...],state_predictive_posterior


### Compute log policy posteriors --------------------------------------------------------------------------------
def policy_posterior(rngkey,current_timestep,Th,filter_end_of_trial,
                     qs,vecA,vecB,vecC,vecE,
                     vecA_novel,vecB_novel,
                     planning_options):
    Np = vecB.shape[-1]
    Ns = qs.shape[-1]
    
    # Extract all the options from the planning_options
    efe_compute_a_nov = planning_options["a_novelty"]
    efe_compute_b_nov = planning_options["b_novelty"]
    old_efe_computation = planning_options["old_novelty_computation"]
    
    N_action_branches = planning_options["N_action_branches"] 
        # Should the 1st always be Np ?
    action_sample_temperature = planning_options["plantree_gumbel_action_sample_temp"]
    
    N_state_branches = planning_options["N_state_branches"]
    state_sample_temperature = planning_options["plantree_gumbel_state_sample_temp"]
    
    explore_remaining_paths = planning_options["explore_joint_remaining"]
        # When Sh action paths have been explored, do we also explore the remaining
        # joint state distribution as a last branch ?
        
        
    # _______________________________________________________________________________________________________
    # Checking the tree architecture : the tree structure is static and is constrained by the (Np,Ns) system
    # This should be implemented by an encompassing class
    N_state_branches = min(Ns,N_state_branches)  # Sh cannot be bigger than Ns
    if N_state_branches==0:
        explore_remaining_paths = True
    if (N_state_branches >= Ns - 1) and explore_remaining_paths:
            # Can't explore remaining paths if they are already all explored :)
        N_state_branches = Ns
        explore_remaining_paths = False
        
    N_action_branches = max(1,min(Np,N_action_branches))  # Ph cannot be bigger than Np or smaller than 1
    # _______________________________________________________________________________________________________
    
    prep_qs = jnp.expand_dims(qs,axis=-2)
    EFE_per_action,last_action_posterior,predictive_state_posterior = compute_EFE(rngkey,prep_qs,current_timestep,
            vecA,vecB,vecC,vecE,
            vecA_novel,vecB_novel,
            filter_end_of_trial,
            Th,
            N_action_branches,action_sample_temperature,
            N_state_branches,state_sample_temperature,explore_remaining_paths,
            option_a_nov=efe_compute_a_nov,option_b_nov=efe_compute_b_nov,
            additional_options_planning=old_efe_computation)

    return EFE_per_action,last_action_posterior
