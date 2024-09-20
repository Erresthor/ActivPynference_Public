import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax
from jax import vmap
from functools import partial


# CROSS ACTION GENERALIZATION ___________________________________________________________________________________
# Action space mapping :
def get_generalizing_table(_transition_indexes,_structure):
    """_summary_

    Args:
        _transition_indexes (_type_): _description_. The order of these indexes matters
                        when ordering the cartesian action space !
        _structure (_type_): "radial" or "cartesian"

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: a scalar table based on the closeness of each actions.
    """
    
    _Nu = _transition_indexes.shape[0]
    generalizing_table = np.zeros((_Nu,_Nu))
        
    if _structure == "radial":        
        radial_subdivision = 2*np.pi/_Nu
        
        for source_action in _transition_indexes:
            for other_action in _transition_indexes :
                angle = other_action*radial_subdivision-source_action*radial_subdivision
                
                generalizing_table[other_action,source_action] = np.cos(angle)
    
    elif _structure == "cartesian":
        # Assuming that the action structure is square : 
        Nu_x = np.sqrt(_Nu).astype(int)
            
        # assert type(Nu_x) == int,"Number of actions does not match a cartesian action space structure"

        action_in_struct = np.reshape(_transition_indexes,(Nu_x,Nu_x))        
        middle_point_index = np.array(((Nu_x-1.0)/2.0,(Nu_x-1.0)/2.0))
        
        for source_index, source_action in np.ndenumerate(action_in_struct):
            source_vector = (np.array(source_index) - middle_point_index)
            norm = np.linalg.norm(source_vector)+1e-10
            for other_index, other_action in np.ndenumerate(action_in_struct) :
                other_vector = (np.array(other_index) - middle_point_index)
                
                dot_prod = np.dot(source_vector/norm,other_vector/norm)

                generalizing_table[other_action,source_action] = dot_prod
    else :
        raise NotImplementedError("This action structure was not implemented : {}".format(_structure))

    return generalizing_table * (1.0 - np.eye(_Nu)) + np.eye(_Nu)

# ENCODING A NON INTEGER scalar_value TO A VECTOR : 
def float_oh(scalar_value,Ns):
    lb = jnp.floor(scalar_value+1e-10)
    
    lb_density = 1.0 - (scalar_value - lb)
    
    ub_density = 1.0 - lb_density
    
    return lb_density*jax.nn.one_hot(lb,Ns) + ub_density*jax.nn.one_hot(lb+1,Ns)

def generalize_transition(from_state_i,to_state_j,effect_weight,Ns,clip=True):
    effect_of_action = to_state_j - from_state_i
    
    # This action effect can be affected by the action weight : 
    weighted_effect_of_action = effect_weight*effect_of_action
    
    # The transition is changed for this effect_weight. 
    # The from_state stays the same, but the to_state is now :
    new_to_state = from_state_i + weighted_effect_of_action
    if clip :
        new_to_state = jnp.clip(new_to_state,0.0,Ns-1)
    
    vec_new_to_state = float_oh(new_to_state,Ns)
    return new_to_state,vec_new_to_state


# Depending on the generalization weight of a specific action
# what is the transition mapping ?
def transition_mapping_depending_on_state(effect_weight,Ns,clip=True):
    from_states = jnp.arange(Ns)
    to_states = jnp.arange(Ns)
    
    action_effect_static = partial(generalize_transition,effect_weight=effect_weight,Ns=Ns)
    
    # Mapped across origin and subsequent states
    mapped_transition_builder = vmap(vmap(action_effect_static,in_axes=(0,None)),in_axes=(None,0))
    
    _mapping_scalar,_mapping_vec = mapped_transition_builder(from_states,to_states)
    
    return _mapping_scalar,_mapping_vec                                                                                                                                         

def generalize_across_actions(db_all_timesteps,gen_table,
                              return_extrapolated_only = True,clip=True):
    Ns,Ns,Nu = db_all_timesteps.shape

    expected_state_for_every_action,transition_mapping_for_every_action = vmap(vmap(lambda x : transition_mapping_depending_on_state(x,Ns,clip)))(gen_table)
    
    gen_action_db = jnp.einsum("uvijk,ijv->kju",transition_mapping_for_every_action,db_all_timesteps)
    
    if return_extrapolated_only:
        gen_action_db = gen_action_db - db_all_timesteps
    
    return gen_action_db
    
    
# CROSS STATES GENERALIZATION ___________________________________________________________________________________

# To generalize action mappings in linear state spaces
def weighted_padded_roll(matrix,generalize_fadeout):
    assert matrix.ndim == 2,"Weighted Padded Roll only implemented for 2D arrays"
    K = matrix.shape[0]
    roll_limit = K
    
    padded_matrix = jnp.pad(matrix,((K,K),(K,K)),mode="constant",constant_values=0)
     
    rolling_func = lambda k : jnp.roll(padded_matrix,k,[-1,-2])*generalize_fadeout(jnp.abs(k))
    
    all_rolled = vmap(rolling_func)(jnp.arange(-roll_limit,roll_limit+1))
    
    # Remove padding : 
    all_rolled = all_rolled[...,K:-K,K:-K]
    
    new_db = all_rolled.sum(axis=-3)
    
    return new_db


def generalize_across_states(db_all_timesteps,generalize_function,
                            return_extrapolated_only = True):
    
    # db is the history of state transitions for each action and at each timestep
    # Making some broad hypotheses about the structure of the latent state space,
    # we can generalize the findings at a coordinate $(s_{t+1},s_t)$ to over states
    # This can be done by rolling the db_all_timesteps matrix across s_(t+1) and s_t axes
    # simultaneously : 
    gen_states_db = vmap(lambda bu : weighted_padded_roll(bu,generalize_function),in_axes=(-1))(db_all_timesteps)
    gen_states_db = jnp.moveaxis(gen_states_db,0,-1)
    
    if return_extrapolated_only :
        return gen_states_db - db_all_timesteps
    
    return gen_states_db


def extrapolate_deltab(db_all_timesteps,
                       generalize_state_function=None,
                       generalize_action_table=None,cross_action_extrapolation_coeff=0.1,
                       option_clip=False):    
    # Generalization : we assume that evidence for a specific transition also provides information regarding
    # - The same relative transition but in other states (generalization across states)
    # - A related transition for another action (generalization across actions)
    # db_all_timesteps is the history of state transitions for each action
    # Making some broad hypotheses about the structure of the latent state space,
    # we can generalize the findings at a coordinate $(s_{t+1},s_t)$ to over states & actions
    # This can be done by rolling the db_all_timesteps matrix across s_(t+1) and s_t axes
    # simultaneously : 

    if not(generalize_action_table is None):
        extrapolated_other_actions = generalize_across_actions(db_all_timesteps,generalize_action_table,
                                            return_extrapolated_only=True,clip=option_clip)
        db_all_timesteps = db_all_timesteps + cross_action_extrapolation_coeff*extrapolated_other_actions
    
    if not(generalize_state_function is None):
        extrapolated_other_states = generalize_across_states(db_all_timesteps,generalize_state_function,
                                                             return_extrapolated_only=True)
        
        db_all_timesteps = db_all_timesteps + extrapolated_other_states
    
    return db_all_timesteps
    
    
#     
#     db_gen_across_states = generalize_across_states(db_all_timesteps,generalize_state_function)
    
    
    
#     return pb + lr_b*gen_db
