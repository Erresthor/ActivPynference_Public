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

from .jax_toolbox import _normalize,_jaxlog
from .planning_tools import compute_Gt_array,compute_novelty
from .layer_infer_state import compute_state_posterior

# A set of functions for agents to plan their next moves.  Note that we use a bruteforce treesearch approach, which is
# a computational nightmare, but may be adapted to fit short term path planning.

# _________________________________________________________________________________________
# Functions used to build the policy tree. 
#  --> Add a level to the tree (changing a tensor of shape K to shape K x (Ph x Sh))
#  --> Evaluate the EFE at a specific level of the tree

# Vectorize this along all tree nodes for a specific explored future tmstp
@partial(jit,static_argnames=["option_a_nov","option_b_nov",'additional_options_planning'])
def compute_efe_node(t,
            previous_posterior,
            vecA,vecB,vecC,vecE,
            vecA_nov,vecB_nov,
            trial_end_scalar,   
            option_a_nov=True,option_b_nov=False,
            additional_options_planning=False):
    # For :
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

def path_heuristic(visited_state_densities,visited_efes):
    """Returns a scalar value that is used as an heuristic to decide if this path should be
    explored further or not. Higher values promise higher neg EFE and are thus more sought-after.
    
    Note : this path may have a low probability of actually happening. This is encoded in the
    visited_densities part of the tree. 
    Each path has its own "promised negative EFE" (= sum of the visited neg-EFEs)
    and the probability of actually coming through with it = prod(p_i)
    
    
    The bigger result is the most interesting, but importantly, the agent has to penalize 
    low probability outcomes because they may not be that interesting.
    
    
    Args:
        visited_state_densities (_type_): _description_
        visited_efes (_type_): a set of T negative EFE values
                ( the higher, the better !). 
                
            

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """



# @partial(jit,static_argnames=["Th","path_cap","path_compute_cap","option_a_nov",'option_b_nov','additional_options_planning'])
def compute_EFE(initial_qs,initial_t,
        vecA,vecB,vecC,vecE,
        nov_a,nov_b,
        filter_trial_end,
        Th,path_cap=None,path_compute_cap=None,
        option_a_nov=True,option_b_nov=False,
        additional_options_planning=False):
    
    # Initialize the needed weights :
    EFE_FLOOR = -10000
    Ns = initial_qs.shape[-1]
    Np = vecB.shape[-1]
    
    there_is_no_provided_cap = ((path_compute_cap == None)and(path_cap==None))
    if (there_is_no_provided_cap):
        raise RuntimeError("Can't use capped learning without a cap ! Please specify a path or a compute cap.")
    
    K = path_cap
    if not(path_compute_cap == None):
        K = path_compute_cap/(Ns*Np)
        pass # The cap affects the amount of computations performed
    
    individual_actions_extractor = vmap(lambda x : jax.nn.one_hot(x,Np))(jnp.arange(Np))
    individual_state_extractor = vmap(lambda x : jax.nn.one_hot(x,Ns))(jnp.arange(Ns))
    
    
    def next_posterior(_explored_state,_pred_post):
        # Predictive observation based on explored_state realization
        po = tree_map(lambda a_m : jnp.einsum('oi,i->o',a_m,_explored_state),vecA)
        # Use it to compute the expected posterior if that happens : 
        qs,F = compute_state_posterior(_pred_post,po,vecA)
        return qs
    vect_compute_next_posterior = vmap(vmap(vmap(next_posterior, in_axes=(0,None))))
        # This vectorized function will help us later :D
        
        
    compute_efe_func = (lambda _qs,_t,_filt : compute_efe_node(_t,_qs,
                                            vecA,vecB,vecC,vecE,
                                            nov_a,nov_b,
                                            _filt, # Are we at the last timestep ?
                                            option_a_nov,option_b_nov,additional_options_planning))    
    vect_compute_efe = vmap(compute_efe_func,in_axes=(0,None,None))
        
    
    # Tree building : 
    def scanner(carry,x):
        (previous_posteriors
            ,sum_of_path_Gs,sum_of_path_logliks) = carry
        (explored_t,filter_trial_horizon) = x
        
        # previous_posteriors is a tensor with size [K x Ns]
        
        # Expand here : 
        # From K to Ns x Nu x K
        # 1. Effect of actions
        _next_priors = vmap(lambda _p : jnp.einsum("iju,kj,eu->eki",vecB,previous_posteriors,individual_actions_extractor))
                        # size [Np x K x Ns], e is the extracted dim in the einsum
        
        # 2. Extract the density of the states explored 
        _next_densities = jnp.einsum("eki,si->sek",_next_priors,individual_state_extractor)
                        # size [Ns x Np x K], s is the extracted dim in the einsum
        # And their one-hot encoding
        _next_states_explored = jnp.einsum("eki,sj->sekj",_next_priors,individual_state_extractor)
                        # size [Ns x Np x K x Ns] 
                        
                
        # We evaluate next_posterior for :
        # + _next_priors along dimensions 1,2
        # + _next_states_explorex along dimensions 0,1,2
        _next_posteriors = vect_compute_next_posterior(_next_states_explored,_next_priors)
                        # size [Ns x Np x K x Ns] 
        
        # The branches are created and their parameters noted. Now, let's flatten them :
        # (here, -1 = K' = K x Ns x Np)
        _next_densities = jnp.reshape(_next_densities,(-1,))
        _next_states_explored = jnp.reshape(_next_states_explored,(-1,Ns))
        _next_posteriors = jnp.reshape(_next_states_explored,(-1,Ns))
        
        
        # compute the efe for all these new branches ! (no flattening here, avoid implicit shape changes)
        _next_efe = vect_compute_efe(_next_posteriors,explored_t,filter_trial_horizon)
                        # size [K' x Np] 
        
        # Nice , we've got the quantities needed. Now, we need to pick the 
        # ones that we will keep exploring and the ones we will give up !
        
        
        # Dynamic tree pruning : remove the paths that a) seem implausible and b) have low efe
        # until we get back to the cap K
        # marginalized sum EFE : = sum(log (qs_t) * G_t) ? (this does )
        heuristic = a_nice_function(path_efe_history,path_prob_history)
            # A [K']-sized tensor that we may sort to pick relevant action-state paths
        keep_those = jnp.argsort(heuristic)[:K]
        
        carry_next = (sum_of_path_Gs,sum_of_path_logliks)
        
        return carry_next
    
    

@partial(jit,static_argnames=["Th","Sh","Ph","option_a_nov",'option_b_nov','additional_options_planning','explore_remaining_paths'])
def compute_EFE_old(qs_current,start_t,
        vecA,vecB,vecC,vecE,
        nov_a,nov_b,
        filter_trial_end,
        Th,Sh,Ph,
        option_a_nov=True,option_b_nov=False,
        additional_options_planning=False,explore_remaining_paths=False):
    EFE_FLOOR = -10000
    Ns = qs_current.shape[-1]
    Np = vecB.shape[-1]
    
    # Utils -------------------------------------------------------- 
    # These are the predicted shapes for the next timestep
    # We only use the computed EFE here !
    if explore_remaining_paths:
        exploration_step_shape = (Ph,Sh+1)
    else : 
        exploration_step_shape = (Ph,Sh)
    
    
    # Forward tree building ----------------------------------------
    # 2 philosophies for this problem : 
    # 1. We want to use scan, and therefore need a constant shape for each prospective step
    #      thus we precompute an array of paths we want to explore, and then scan them
    # 2. We want to reduce redundant computation and therefore have an expanding tree. 
    #      it seems that using lists (or maybe pytrees ?) works best in that case
    # 3. We want to use scan WITHOUT predefining the explored action paths. Let's introduce a 
    #      planning algorithm that continuously prunes the potential new paths explored.
    #      This needs to be further refined (what is the minimized quantity).
    #      For now, we'll use a mix of state log likelihood and path negative EFE.
    # 
    # This function uses an application of 3.
    
    # First timestep EFE computation
    qs_pi = qs_current
        
    # qs_pi is K x [Ns]
    compute_node_func = (lambda _qs : compute_efe_node(start_t,_qs,
                                            vecA,vecB,vecC,vecE,
                                            nov_a,nov_b,
                                            filter_trial_end[0], # Are we at the last timestep ?
                                            option_a_nov,option_b_nov,additional_options_planning))
    qs_next_initial,efe_next_actions = vmap(compute_node_func)(qs_pi)
    exploration_tree = [[qs_pi,efe_next_actions,None,None]]
        # Predictive priors & efe for this timestep
        #      K x  [Np x Ns]  |  K x [Np]
    
    
    # The following steps will be unrolled by the compiler ! 
    # We expand the tree repeateadly until we reach the desired temporal horizon
    # Autobots, roll out !
    qs_previous = qs_pi  # This posterior will also be used for subsequent estimations
    N_efe_computed_history = []
    for explorative_timestep in range(1,Th+1): 
        t = start_t+explorative_timestep # t in [start_t+1, current_t+Th]
                      
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
                                            filter_trial_end[explorative_timestep],  # If we are at trial end, this is equivalent to the habits vecE
                                            option_a_nov,option_b_nov,additional_options_planning))
        qs_next,efe_next_actions = vmap(compute_node_func)(qs_pi) 
        
        efe_next_actions = efe_next_actions*filter_trial_end[explorative_timestep-1]  
                # If the previous timestep is trial end or after, this computation is not taken into account 
                # This is redundant, can we just remove it ?
                
        N_efe_computed_history.append(efe_next_actions.shape[0])
        
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
    
    # Backward tree summing ----------------------------------------
    def remerge_action(_efe_children,_children_ut):
        """ 
        Map a K x Ph tensor of EFE onto a K x Np space using a K x Ph x Np mapping rule.
        Unexplored action paths should have a very low EFE !
        """ 
        carry_efe = jnp.einsum("abc,ab->ac",_children_ut,_efe_children)  
                # Efe of subsequent next steps for the explored action paths
                # _children_ut has shape Nbranches x Ph x Np
                
        # What are the unexplored actions ?
        explored_filter = _children_ut.sum(axis=-2)  
                            # A (Nbranches x Np) tensor with 1.0 where we explored an action
                            # and 0 where we did not
        unexplored_filter = 1.0 - explored_filter
        
        unexplored_efe = unexplored_filter * EFE_FLOOR
                    # If the path was not explored, we may assume that the EFE is very high :(
        
        return carry_efe + unexplored_efe # jnp.where(carry_efe==0,EFE_FLOOR,carry_efe)
    
    
    # This will be unrolled ! (needs to be done sequentially, 
    # big Th values are obviously discouraged)
    # Autobots, roll out !       
    carry_efe = jnp.zeros_like(exploration_tree[Th][1])
    for explorative_timestep in range(Th,0,-1): # Th -> Th-1 -> ... -> 1
        space_tuple_next = (exploration_step_shape*(explorative_timestep))

        # These are the predicted values for the next timestep
        # We only use the computed EFE here !
        [qs_tsmtp,efe_this_tsmtp,state_branch_densities,ut_next_tsmtp] = exploration_tree[explorative_timestep]
        
        state_branch_densities = jnp.reshape(state_branch_densities,space_tuple_next)
        
        efe_this_tsmtp = jnp.reshape(efe_this_tsmtp + carry_efe,space_tuple_next+(Np,))
                   
        # We marginalize the efe for the next timestep across expected actions ... 
        # (should there be a precison parameter here ?)
        margin_efe = jnp.sum(efe_this_tsmtp*jax.nn.softmax(efe_this_tsmtp,axis=-1),axis=-1)
        # ... and states
        margin_efe_next_tmstp = jnp.sum(state_branch_densities*margin_efe,axis=-1)

        # To get a quantity that can be added to the (previous) explorative timestep, it
        # has to map to the policy axis. To do this, we use our history of the explored action !
        # The unexplored actions should have EFE = -inf.
        flattened_margin_efe = jnp.reshape(margin_efe_next_tmstp,(-1,Ph))
                    # Last dim shape is Ph
        
        carry_efe = remerge_action(flattened_margin_efe,ut_next_tsmtp)
                    # Last dim shape is Np
    
    final_efe = carry_efe+exploration_tree[0][1]
    
    # predictive posterior over the very next hidden state given this posterior : 
    u_post = jax.nn.softmax(final_efe,axis=-1)
    
    state_predictive_posterior = jnp.einsum("aus,au->s",qs_next_initial,u_post)
    
    return final_efe[0,...],u_post[0,...],state_predictive_posterior,N_efe_computed_history
    

### Compute log policy posteriors --------------------------------------------------------------------------------
# @partial(jit, static_argnames=['Np','Th','gamma'])
def policy_posterior(current_timestep,Th,filter_end_of_trial,
                     qs,vecA,vecB,vecC,vecE,vecA_novel,vecB_novel,
                     planning_options):
    Np = vecB.shape[-1]
    Ns = qs.shape[-1]
    
    # Extract all the options from the planning_options
    efe_compute_a_nov = planning_options["a_novelty"]
    efe_compute_b_nov = planning_options["b_novelty"]
    other_option = planning_options["old_novelty_computation"]
    
    Ph = planning_options["plantree_action_horizon"]
    
    Sh = planning_options["plantree_state_horizon"]
    explore_remaining_paths = planning_options["explore_joint_remaining"]
        # When Sh action paths have been explored, do we also explore the remaining
        # joint state distribution as a last branch ?
        
        
    # _______________________________________________________________________________________________________
    # Checking the tree architecture : the tree structure is static and is constrained by the (Np,Ns) system
    # This should be implemented by an encompassing class
    Sh = min(Ns,Sh)  # Sh cannot be bigger than Ns
    if Sh==0:
        explore_remaining_paths = True
    if (Sh >= Ns - 1)and explore_remaining_paths:
            # Can't explore remaining paths if they are already all explored :)
        Sh = Ns
        explore_remaining_paths = False
        
    Ph = max(1,min(Np,Ph))  # Ph cannot be bigger than Np or smaller than 1
    # _______________________________________________________________________________________________________
    
    prep_qs = jnp.expand_dims(qs,axis=-2)
    EFE_per_action,last_action_posterior,predictive_state_posterior,N_efe_computed_history = compute_EFE(prep_qs,current_timestep,
                    vecA,vecB,vecC,vecE,
                    vecA_novel,vecB_novel,
                    filter_end_of_trial,
                    Th,Sh,Ph,
                    option_a_nov=efe_compute_a_nov,option_b_nov=efe_compute_b_nov,
                    additional_options_planning=other_option,explore_remaining_paths=explore_remaining_paths)

    return EFE_per_action,last_action_posterior
