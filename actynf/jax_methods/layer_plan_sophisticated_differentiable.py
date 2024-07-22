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

# "---------------------------------------"

def branching_out_actions(efe_each_action,N_action_branches,
                         _action_sample_temperature = 0.001):
    Np = efe_each_action.shape[-1]
    
    # TODO : We should use a Gumbel-Softmax trick here to avoid 
    # split actions !   
    
    # Here, we add a deterministic biais to avoid equal value actions :
    # (in case the EFE for two actions is exactly equal)
    EPSILON = 1e-8
    BIAIS = jnp.arange(Np)*EPSILON # [0.0, EPSILON/(Np-1),..., EPSILON]
    biaised_efe = efe_each_action + BIAIS
    
    def action_brancher(carry,xs):
        remaining_distribution = carry
        
        branch_explored_state_distribution = jax.nn.softmax(remaining_distribution/_action_sample_temperature)
            # A (nearly) one-hot encoding of the explored state branch
        
        # explored_density = (branch_explored_state_distribution*_pred_prior).sum()
        explored_density = (branch_explored_state_distribution*remaining_distribution).sum()
            # a scalar estimate of the total density predicted by this path
        
        remaining_distribution = jax.nn.relu(remaining_distribution - branch_explored_state_distribution*explored_density)
        
        return remaining_distribution,(branch_explored_state_distribution,explored_density)
    
    # Distribution over what actions we may explore, to be further branched !
    init_distribution_scan = jax.nn.softmax(biaised_efe)
    print(init_distribution_scan)
    unexplored_remainder,(explored_actions,explored_action_densities) = jax.lax.scan(action_brancher,init_distribution_scan,jnp.arange(N_action_branches))
    print(explored_action_densities)
    # No remainder computations here : the unexplored actions are assumed to have very low (neg)EFE.
    return explored_actions
    
    

@partial(jit,static_argnames=["N_state_branches","_state_remainder_branch"])
def branching_out_states(_pred_prior,N_state_branches,
                         _state_remainder_branch=True,
                         _state_sample_temperature = 0.01):       
    """
    Decompose each predictive priors into Sh individual states !
    If remainder_state is set to True, also group the remaining unexplored states
    together and compute its EFE too !
    TODO : make a version of this that uses OTT : 
    https://ott-jax.readthedocs.io/en/latest/tutorials/soft_sort.html

    Args:
        _pred_prior (_type_): _description_
        N_state_branches (_type_): _description_
        _state_remainder_branch (bool, optional): _description_. Defaults to True.
        _state_sample_tempoerature (float, optional): _description_. Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    EPSILON = 1e-10
    Ns, = _pred_prior.shape

    # TODO : We should use a Gumbel-Softmax trick here to avoid 
    # split states !
    
    # Decompose the probability distribution into individual realizations :
    
    
    def state_brancher(carry,xs):
        remaining_distribution = carry
        
        branch_explored_state_distribution = jax.nn.softmax(remaining_distribution/_state_sample_temperature)
            # A one-hot encoding of the explored state branch
        
        # explored_density = (branch_explored_state_distribution*_pred_prior).sum()
        explored_density = (branch_explored_state_distribution*remaining_distribution).sum()
            # a scalar estimate of the total density predicted by this path
        
        remaining_distribution = jax.nn.relu(remaining_distribution - branch_explored_state_distribution*explored_density)
        
        return remaining_distribution,(branch_explored_state_distribution,explored_density)
        
        
    init_distribution_scan = _pred_prior
    unexplored_remainder,(explored_states,explored_densities) = jax.lax.scan(state_brancher,init_distribution_scan,jnp.arange(N_state_branches))
    
    
    if _state_remainder_branch:
        # This was the total distibution space left unexplored :
        norm_remainder,_ = _normalize(unexplored_remainder+EPSILON)
            # The epsilon term is to avoid empty distributions for
            # very low temperatures
            # TODO : correct the _normalize definition ?
        last_explored_state = jnp.expand_dims(norm_remainder,axis=-2)
        
        remainder_density = jnp.array([unexplored_remainder.sum()])


        # Add this to the previously explored paths
        explored_states = jnp.concatenate([explored_states,last_explored_state],axis=-2)
        explored_densities = jnp.concatenate([explored_densities,remainder_density],axis=-1)
    
    explored_densities,_ = _normalize(explored_densities) # Explored densities should sum to 1
    
    return explored_states,explored_densities

@jit
def branch_predictive_posterior(_predictive_prior,_branch_state_outcome,vecA):
    # Compute the expected observations under this state outcome
    # and how it will affect our posterior beliefs
    
    # Predictive observation based on explored_state realization
    po = tree_map(lambda a_m : jnp.einsum('oi,i->o',a_m,_branch_state_outcome),vecA)
    
    # Use it to compute the expected posterior if that happens : 
    qs,F = compute_state_posterior(_predictive_prior,po,vecA)
    
    return qs



@partial(jit,static_argnames=["Th","Sh","Ph","option_a_nov",'option_b_nov','additional_options_planning','explore_remaining_paths'])
def compute_EFE(qs_current,start_t,
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
    # 
    # This function uses an application of 2.

    exploration_tree = []  # We will fill this tree with Th levels of branches
    
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
    

@partial(jit,static_argnames=["Sh","option_a_nov",'option_b_nov','additional_options_planning','explore_remaining_paths'])
def diff_efe(initial_posterior,qpi,
                vecA,vecB,vecC,vecE,
                nov_a,nov_b,
                Sh,
                option_a_nov=True,option_b_nov=False,
                additional_options_planning=False,explore_remaining_paths=True,
                tau_branch_split = 0.05):
    """
    Planning without an explicit action tree !
    Using a tensor of action probabilities from current_t to current_t+Th,
    compute the expected free energy.
    
    This function is meant to be differentiated to be minimized. 
    Intuition : differentiating w.r.t. later /earlier slices of qpi will lead to 
    different ways of perceiving the future ! 

    Args:
        qpi (_type_): _description_
    """
    Th,Np = qpi.shape
    Ns, = initial_posterior.shape
    
    # Utils -------------------------------------------------------- 
    # These are the predicted shapes for the next timestep
    # We only use the computed EFE here !
    if explore_remaining_paths:
        exploration_step_size = Sh+1
    else :
        exploration_step_size = Sh
        
    
    state_branching_func =  vmap(partial(branching_out_states, N_state_branches=Sh,
                                    _state_remainder_branch=explore_remaining_paths,
                                   _state_sample_temperature = tau_branch_split))
    state_belief_update_func = vmap(vmap(partial(branch_predictive_posterior,vecA=vecA),in_axes=(None,0)))
    
    
    
    
    
    efe_compute_func = partial(compute_efe_action_posterior,vecA=vecA,vecB=vecB,vecC=vecC,vecE=vecE,
                                    vecA_nov=nov_a,vecB_nov=nov_b,  
                                    option_a_nov=option_a_nov,option_b_nov=option_b_nov,
                                    additional_options_planning=additional_options_planning)
    mapped_efe_compute = vmap(efe_compute_func,in_axes=[None,0,None,None])
        # This function will get mapped accross previous posterior and predicted posteriors
    
    initial_posterior = jnp.expand_dims(initial_posterior,-2)
    
    trial_end_scalar = 1.0
    previous_prior,initial_efe = mapped_efe_compute(qpi[0],initial_posterior,0,trial_end_scalar)
    
    # Tree building
    density_tree = [jnp.array([1.0])]
    efe_tree = [initial_efe]
    
    for t in range(1,Th):
        trial_end_scalar = 1.0
        # previous_prior is a tensor of shape K x Ns, with K = (Sh+1)^t        
        
        # Split into Sh branches accounting for each potential outcome:
        branch_outcomes,branch_densities =state_branching_func(previous_prior)
            # Branched states : [K x (Sh(+1)?) x Ns] 
            # branched densities : [K x (Sh(+1)?)] 
        
        # Each branched states generates a predicted observation, that the agent
        # expects will change its beliefs !
        branch_predictive_posteriors = state_belief_update_func(previous_prior,branch_outcomes)
            # branch_predictive_posteriors : [K x (Sh(+1)?) x Ns] 
        
        branch_densities = jnp.reshape(branch_densities,(-1,))
        branch_posterior = jnp.reshape(branch_predictive_posteriors,(-1,Ns))        
                            # a [K' x Ns] tensor of predictive posteriors
                            # K' = K x (Sh (+1?))
        
        # Compute EFE and predicted priors for next action for each potential posterior
        trial_end_scalar = 1.0
        next_prior,efe = mapped_efe_compute(qpi[t],branch_posterior,t,trial_end_scalar)
            # Branched priors : [K' x Ns] 
                    
        density_tree.append(branch_densities)
        efe_tree.append(efe)
    
        previous_prior = next_prior
    
    
    # # Tree collapsing :
    # # Compute the EFE under the possible states seen here !
    # efe_carry = efe_tree[Th-1]
    carry_efe = jnp.zeros_like(efe_tree[Th-1])
    for t in range(Th-1,0,-1):
        densities = jnp.reshape(density_tree[t],(-1,exploration_step_size))
            # marginalized EFEs depend on outcomes for timesteps > 1
        
        efes = jnp.reshape(efe_tree[t]+carry_efe,(-1,exploration_step_size))
            # The EFE also comprises the subsequent timesteps
        marginalized_efe = (efes*densities).sum(axis=-1)
        
        carry_efe = marginalized_efe
    
    total_efe = carry_efe+efe_tree[0]*density_tree[0]
    return total_efe[0]
  

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
