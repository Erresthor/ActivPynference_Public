# _summary_
# A set of dictionnaries (and the functions to make them)
# that are used by our layer as options for planning / learning (/action picking ?)

def get_planning_options(
                        Th,planning_method = "sophisticated",
                        state_horizon = 2,action_horizon=5,explore_remaining_paths=True,
                        a_novel=True,b_novel=False,
                        old_a_novel=True):
    return {
        "horizon" : Th,
        "a_novelty" : a_novel,
        "b_novelty" : b_novel,
        "old_novelty_computation" : old_a_novel,
        "method": planning_method,  # "full_tree" or "joint_tree"
        "plantree_action_horizon":action_horizon,
        "plantree_state_horizon":state_horizon,
        "explore_joint_remaining":explore_remaining_paths
    }


def get_learning_options(learn_a = False,learn_b=False,learn_d=False,lr_a=1.0,lr_b=1.0,lr_d=1.0,run_smoother=False):
    return {
        "bool":{
            "a":learn_a,
            "b":learn_b,
            "c":False,  # Not yet implemented
            "d":learn_d,
            "e":False   # Not yet implemented
        },
        "rates":{
            "a":lr_a,
            "b":lr_b,
            "c":0.0,
            "d":lr_d,
            "e":0.0
        },
        "smooth_states":run_smoother
    }

def get_action_selection_options(selection_method="stochastic",alpha = 16,gamma=None,):
    return {
        "alpha":alpha,
        'gamma':gamma,
        "method":selection_method
    }

# Default temporal horizon is the minimum (2)
DEFAULT_PLANNING_OPTIONS = get_planning_options(2)

DEFAULT_LEARNING_OPTIONS = get_learning_options()

DEFAULT_ACTION_SELECTION_OPTIONS = get_action_selection_options()