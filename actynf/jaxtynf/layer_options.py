import jax.numpy as jnp
# _summary_
# A set of dictionnaries (and the functions to make them)
# that are used by our layer as options for planning / learning (/action picking ?)

def get_planning_options(
                        Th,planning_method = "sophisticated",
                        state_horizon = 2,action_horizon=5,explore_remaining_paths=True,
                        a_novel=True,b_novel=False,
                        old_efe_computation=True):
    return {
        "horizon" : Th,
        
        "a_novelty" : a_novel,
        "b_novelty" : b_novel,
        "old_novelty_computation" : old_efe_computation,
        
        "method": planning_method,  # for now, only "classic" or "sophisticated" are supported
        
        "plantree_action_horizon":action_horizon,
        "plantree_state_horizon":state_horizon,
        "explore_joint_remaining":explore_remaining_paths
    }


def get_learning_options(learn_a = False,learn_b=False,learn_d=False,learn_e=False,
                         lr_a=1.0,lr_b=1.0,lr_d=1.0,lr_e=1.0,
                         fr_a=0.0,fr_b=0.0,fr_d=0.0,fr_e=0.0,
                         method="vanilla+backwards",
                         state_generalize_function = None,
                         action_generalize_table = None,
                         cross_action_extrapolation_coeff = 0.0,em_iterations=4):
    return {
        "method":method, # "vanilla+backwards","vanilla" or "em"
        "bool":{
            "a":learn_a,
            "b":learn_b,
            "c":False,  # Not yet implemented
            "d":learn_d,
            "e":learn_e
        },
        "learning_rates":{
            "a":lr_a,
            "b":lr_b,
            "c":0.0,
            "d":lr_d,
            "e":lr_e
        },
        "forgetting_rates":{
            "a":fr_a,
            "b":fr_b,
            "c":0.0,
            "d":fr_d,
            "e":fr_e
        },
        "state_generalize_function" : state_generalize_function,
        "action_generalize_table" : action_generalize_table,
        "cross_action_extrapolation_coeff" : cross_action_extrapolation_coeff,
        "em_iterations" : em_iterations
    }

def get_action_selection_options(selection_method="stochastic",alpha = 16,gamma=None):
    return {
        "alpha":alpha,
        'gamma':gamma,
        "method":selection_method
    }

# Default temporal horizon is the minimum (2)
DEFAULT_PLANNING_OPTIONS = get_planning_options(2)

DEFAULT_LEARNING_OPTIONS = get_learning_options()

DEFAULT_ACTION_SELECTION_OPTIONS = get_action_selection_options()





