import numpy as np, matplotlib.pyplot as plt, sys,os

# import actynf
# from actynf.base.function_toolbox import normalize
# from actynf.layer.model_layer import mdp_layer
# from actynf.layer.layer_link import establish_layerLink
# from actynf.architecture.network import network

import sys
sys.path.insert(0, '.../')
print(sys.path)
from actynf.base.function_toolbox import normalize
from actynf.layer.model_layer import mdp_layer
from actynf.layer.layer_link import establish_layerLink
from actynf.architecture.network import network

from PIL import Image


def sub2ind(array_shape, sub_tuple):
    x, y = sub_tuple[0],sub_tuple[1]
    if ((x < 0)or(x>=array_shape[0])) or ((y < 0)or(y>=array_shape[1])) :
        raise ValueError(str(sub_tuple) + " is outside the range for array shape " + str(array_shape))
    # Index is x*NumberOfColumns + y
    return x*array_shape[1] + y

def ind2sub(array_shape, ind):
    x = (ind // array_shape[1])
    y = (ind % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return x, y

def get_nf_network(action_selection_temperature,mem_loss):
    T = 10
    # The subject has a certain number of cognitive states , for now, let's just make it a few categorically distinct states 

    # 2d example : states are defined in a 2d grid :
    #               low feedback vals
    # [0,1,2,3]      |
    # [4,5,6,7]      |   This is going "up" in Python terms
    # [8,9,10,11]    \/ 
    # ---------->   high feedback vals
    #
    # this is still going right !
    #
    # But the feedback only convers information for the horizontal dimension :
    Nx = 4
    Ny = 4
    N_cog_states = Nx*Ny  #  Those states may 
    state_grid_list = []
    for x in range(Nx):
        for y in range(Ny):
            state_grid_list.append([x,y])
    vectorized_grid = np.array(state_grid_list)
    # For each state index i, there is a corresponding set of grid coordinates vectorized_grid[i,:]
    

    #-----------------------------------------------------------------------------------------
    # Observations : 
    # 2 modalities of observation : 
    # 1. The external feedback 
    N_outcomes_feedback = 4
    # The true feedback is defined between -1 and 1 depedning on where the subject is on the vertical dimension :
    a1 = np.zeros((N_outcomes_feedback,N_cog_states))
    for state_idx in range(N_cog_states):
        cognitive_coordinates = vectorized_grid[state_idx,:]

        # This can be changed in the future
        distance_to_goal = float(Nx-1 - cognitive_coordinates[0])/(Nx-1) # If you line idx is low, 
                                            # you are far from the goal
        # distance_to_goal = float(cognitive_coordinates[0])/(Nx-1) 
                    # Value between 0 and 1 where is 0 is very far from the goal
                    # and 1 is goal attained

        scale_distance_to_goal = (1.0-distance_to_goal)*(N_outcomes_feedback-1) 
                    # On a scale of 0 to N_outcomes_feedback-1, how close to the goal are we ?
        # print(scale_distance_to_goal)

        scale_distance_to_goal_int = int(scale_distance_to_goal)
        scale_distance_to_goal_float = scale_distance_to_goal - scale_distance_to_goal_int
        # print(scale_distance_to_goal_int,scale_distance_to_goal_float)

        lower_idx = scale_distance_to_goal_int
        lower_idx_part = 1.0-scale_distance_to_goal_float
        higher_idx_part = scale_distance_to_goal_float


        if (lower_idx+1 < N_outcomes_feedback):
            a1[lower_idx,state_idx] = lower_idx_part
            a1[lower_idx+1,state_idx] = higher_idx_part
        else : 
            a1[lower_idx,state_idx] = 1.0

    # # Plot the preference map
    # show_this = np.zeros((Nx,Ny))
    # for state_idx in range(N_cog_states):
    #     x,y = ind2sub((Nx,Ny),state_idx)

    #     show_this[x,y]= sum([a1[i,state_idx]*i for i in range(N_outcomes_feedback)])
    # print(show_this)
    # image = Image.fromarray((255*show_this).astype(int))
    # resized_pil_image = image.resize((400, 400),resample=Image.NEAREST)
    # resized_pil_image.show()

    # 2. An internal appreciation of each cognitive state
    N_outcomes_internal = N_cog_states
    # This would be a perfect observer :
    a2_true = np.zeros((N_outcomes_internal,N_cog_states))
    for i in range(N_cog_states):
        a2_true[i,i] = 1.0 
    
    # Instead, let's picture an uninformed agent : 
    print(np.round(a1,2))
    a2_subject = 10*normalize(np.ones((N_outcomes_internal,N_cog_states)))

    a_process = [a1,a2_true]
    a_model = [200*a1,a2_subject]

    # Preferences : (JUST linear for now)
    feedback_prefs = np.array(range(N_outcomes_feedback))
    feedback_prefs = np.array([-2,0,5,20])
    print(feedback_prefs)
    c = [feedback_prefs, np.zeros((N_outcomes_internal))]

    #-----------------------------------------------------------------------------------------
    # Initial states (always the same ?)
    d_true = np.zeros(N_cog_states,)
    for y in range(Ny):
        d_true[sub2ind((Nx,Ny),(0,y))] = 1.0
    d_process=[normalize(d_true)]
    d_model = [normalize(d_true)]

    # Transitions : 
    N_useless_actions = 3
    N_go_up = 1
    N_go_down = 1
    N_go_left = 1
    N_go_right = 1
    Nactions = N_go_up + N_go_down + N_go_left + N_go_right + N_useless_actions
    
    U   = [[0,0] for i in range(N_useless_actions)]+[[1,0]*N_go_up]+[[-1,0]*N_go_down]
    U += [[0,1]*N_go_right]+[[0,-1]*N_go_left]
    
    U = np.array(U)
    b1 = np.zeros((N_cog_states,N_cog_states,Nactions))
    for from_x in range(Nx):
        for from_y in range(Ny):
            s = sub2ind((Nx,Ny),(from_x,from_y))
            for u_ix in range(U.shape[0]):
                try :
                    ss = sub2ind((Nx,Ny),(from_x + U[u_ix,0] ,from_y + + U[u_ix,1]))
                    b1[ss,s,u_ix] = 1
                except:
                    b1[s,s,u_ix] = 1
    b = [b1]
    # b_model = [0.5*(5*b1 + np.ones(b1.shape))]
    b_model = [0.01*(2.5*b1+ np.ones(b1.shape))]
    print(b1[:,:,0])

    u = np.array(range(Nactions))
    e = np.ones((Nactions,))

    generative_process = mdp_layer("neurofeedback_generative_process","process",
                    a_process,b,c,d_process,e,u,T,T_horiz=3)

    generative_model = mdp_layer("neurofeedback_subject_model","model",
                    a_model,b_model,c,d_model,e,u,T,T_horiz=3)
    
    # Here, we give a few hyperparameters guiding the beahviour of our agent :
    generative_model.hyperparams.alpha = action_selection_temperature # action precision : 
        # for high values the mouse will always perform the action it perceives as optimal, with very little exploration 
        # towards actions with similar but slightly lower interest

    generative_model.hyperparams.alpha = action_selection_temperature # action precision : 
    # for high values the mouse will always perform the action it perceives as optimal, with very little exploration 
    # towards actions with similar but slightly lower interest
    generative_model.hyperparams.alpha = action_selection_temperature # action precision : 
    # for high values the mouse will always perform the action it perceives as optimal, with very little exploration 
    # towards actions with similar but slightly lower interest

    generative_model.learn_options.eta = 1 # learning rate (shared by all channels : a,b,c,d,e)
    generative_model.learn_options.learn_a = True  # The agent learns the reliability of the clue
    generative_model.learn_options.learn_b = True # The agent does not learn transitions
    generative_model.learn_options.learn_d = True  # The agent has to learn the initial position of the cheese
    generative_model.learn_options.backwards_pass = True  # When learning, the agent will perform a backward pass, using its perception of 
                                               # states in later trials (e.g. I saw that the cheese was on the right at t=3)
                                               # as well as what actions it performed (e.g. and I know that the cheese position has
                                               # not changed between timesteps) to learn more reliable weights (therefore if my clue was
                                               # a right arrow at time = 2, I should memorize that cheese on the right may correlate with
                                               # right arrow in general)
    generative_model.learn_options.memory_loss = mem_loss
                                            # How many trials will be needed to "erase" 50% of the information gathered during one trial
                                            # Used during the learning phase
    generative_model.hyperparams.cap_state_explo = 3
    generative_model.hyperparams.cap_action_explo = 2
    
    establish_layerLink(generative_process,generative_model, [["o.0","o.0"],["o.1","o.1"]])
    
    establish_layerLink(generative_model,generative_process,["u","u"])

    neurofeedback_network = network([generative_process,generative_model],"nf_net")
    # print("Generated the following network :")
    # print(neurofeedback_network.layers[1].e)
    print(generative_model)
    return neurofeedback_network
