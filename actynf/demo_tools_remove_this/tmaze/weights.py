import numpy as np
import matplotlib.pyplot as plt
# import actynf
# from actynf.layer.model_layer import mdp_layer
# from actynf.layer.layer_link import establish_layerLink
# from actynf.architecture.network import network
# from actynf.base.function_toolbox import normalize

from actynf.layer.model_layer import mdp_layer
from actynf.architecture.network import network
from actynf.base.function_toolbox import normalize

import jax.numpy as jnp

# 2 outcome T-maze implementation (we mix cue and position in the exteroceptive outcomes)
# Simplified with no clue quality learning 
def get_T_maze_gen_process(pinit,pHa,pWin):
    """
    pinit : prob of reward initial position being left / right
    pHA : probability of clue giving the correct index position
    pWin : probability of winning if we are in the correct position

    This "new" process generator makes it so that actions 2 & 3 lead to the starting position
    when used from either the left or right part of the T-maze
    In essence, it means that if the mouse has picked a side, it can't
    pick it again (thus preventing the mouse from picking both sides during the 
    same trial !)
    """
    T = 3
    
    #Mapping from states to observed positions & hints
    A_extero = np.zeros(((5,4,2)))
    A_extero[...,0] = np.array([
        [1.0,0.0,0.0,0.0  ], # Position start
        [0.0,1.0,0.0,0.0  ], # Position right
        [0.0,0.0,1.0,0.0  ], # Position left
        [0.0,0.0,0.0,pHa  ], # Cue pointing right
        [0.0,0.0,0.0,1-pHa]  # Cue pointing left
    ])
    A_extero[...,1] = np.array([
        [1.0,0.0,0.0,0.0  ], # Position start
        [0.0,1.0,0.0,0.0  ], # Position right
        [0.0,0.0,1.0,0.0  ], # Position left
        [0.0,0.0,0.0,1-pHa], # Cue pointing right
        [0.0,0.0,0.0,pHa  ]  # Cue pointing left
    ])
    
    # Interoceptive (reward) signals :
    A_intero = np.zeros((3,4,2))
    A_intero[...,0] = np.array([
        [1.0,0.0,0.0,1.0],     # Neutral
        [0.0,pWin,1-pWin,0.0], # Win
        [0.0,1-pWin,pWin,0.0]  # Lose
    ])
    
    A_intero[...,1] = np.array([
        [1.0,0.0,0.0,1.0],     # Neutral
        [0.0,1-pWin,pWin,0.0], # Win
        [0.0,pWin,1-pWin,0.0]  # Lose
    ])                  
    A = [A_extero,A_intero]
    
    
    
    
    # Transition matrixes between hidden states ( = control states)
    
    #1. Transition between behavioural states --> 4 actions
    B_behav_states = np.zeros((4,4,4))
    # - 0 --> Move to start from any state, or stay in the left / right cells if already there
    B_behav_states[:,:,0] = np.array([[1,0,0,1],
                                      [0,1,0,0],
                                      [0,0,1,0],
                                      [0,0,0,0]])
    # - 1 --> Move to choose right from start or hint, else stay where you are
    B_behav_states[:,:,1] = np.array([[0,0,0,0],
                                      [1,1,0,1],
                                      [0,0,1,0],
                                      [0,0,0,0]])
    # - 2 --> Move to choose left from start or hint, else stay where you are
    B_behav_states[:,:,2] = np.array([[0,0,0,0],
                                      [0,1,0,0],
                                      [1,0,1,1],
                                      [0,0,0,0]])  
    # - 3 --> Move to hint from start, else stay where you are
    B_behav_states[:,:,3] = np.array([[0,0,0,0],
                                      [0,1,0,0],
                                      [0,0,1,0],
                                      [1,0,0,1]]) 
    # Note : this verison of the tmaze task implemented in DEM_demo_MDP_XX.m,
    # is different from the usual T-maze task in which the task ends when the agent reaches the goal
    # Here, the agent accumulates the reward / aversive stimulus  across 2 timesteps if it 
    # doesnt visit the hint 
    
    #2. Transition between context states --> The agent cannot act so there is only one :
    B_context_states = np.zeros((2,2,1))
    B_context_states[...,0] = np.eye(2)
    
    B = [B_behav_states,B_context_states]
    
    
    D = [np.array([1.0,0.0,0.0,0.0]),np.array([pinit,1-pinit])]
    
    Ns = [arr.shape[0] for arr in D]
    No = [ai.shape[0] for ai in A]
    
    
    U = np.array([
        [0,0],
        [1,0],
        [2,0],
        [3,0]
    ]).astype(int)

    return A,B,D,U

def get_T_maze_model(pHa,pWin,initial_hint_confidence,la,rs,context_belief):
    """
    pinit : prob of reward initial position being left / right
    pHA : probability of clue giving the correct index position
    pWin : probability of winning if we are in the correct position
    """
    # print("T-maze gen. model set-up ...  ",end='')
    T = 3

    # d = [np.array([128.0,1.0,1.0,1.0]),np.array([2.0,2.0])]
    k_conf = 2.0
    d = [np.array([128.0,1.0,1.0,1.0]),np.array([context_belief*k_conf,(1.0-context_belief)*k_conf])]

        #Mapping from states to observed positions & hints
    A_extero = np.zeros(((5,4,2)))
    v_0 = 200*1.0 # value for known cells
    q_0 = initial_hint_confidence*pHa   # prior belief about clue epistemic value
    q_1 = initial_hint_confidence*(1-pHa)
    A_extero[...,0] = np.array([
        [v_0,0.0,0.0,0.0], # Position start
        [0.0,v_0,0.0,0.0], # Position right
        [0.0,0.0,v_0,0.0], # Position left
        [0.0,0.0,0.0,q_0], # Cue pointing right
        [0.0,0.0,0.0,q_1]  # Cue pointing left
    ])
    A_extero[...,1] = np.array([
        [v_0,0.0,0.0,0.0], # Position start
        [0.0,v_0,0.0,0.0], # Position right
        [0.0,0.0,v_0,0.0], # Position left
        [0.0,0.0,0.0,q_1], # Cue pointing right
        [0.0,0.0,0.0,q_0]  # Cue pointing left
    ])
    
    # Interoceptive (reward) signals :
    A_intero = np.zeros((3,4,2))
    A_intero[...,0] = np.array([
        [1.0,0.0   ,0.0   ,1.0],     # Neutral
        [0.0,pWin  ,1-pWin,0.0], # Win
        [0.0,1-pWin,pWin  ,0.0]  # Lose
    ])*200
    
    A_intero[...,1] = np.array([
        [1.0,0.0,0.0,1.0],     # Neutral
        [0.0,1-pWin,pWin,0.0], # Win
        [0.0,pWin,1-pWin,0.0]  # Lose
    ])*200                  
    a = [A_extero,A_intero]
    
    
    
    #1. Transition between behavioural states --> 4 actions
    B_behav_states = np.zeros((4,4,4))
    # - 0 --> Move to start from any state, or stay in the left / right cells if already there
    B_behav_states[:,:,0] = np.array([[1,0,0,1],
                                      [0,1,0,0],
                                      [0,0,1,0],
                                      [0,0,0,0]])
    # - 1 --> Move to choose right from start or hint, else stay where you are
    B_behav_states[:,:,1] = np.array([[0,0,0,0],
                                      [1,1,0,1],
                                      [0,0,1,0],
                                      [0,0,0,0]])
    # - 2 --> Move to choose left from start or hint, else stay where you are
    B_behav_states[:,:,2] = np.array([[0,0,0,0],
                                      [0,1,0,0],
                                      [1,0,1,1],
                                      [0,0,0,0]])  
    # - 3 --> Move to hint from start, else stay where you are
    B_behav_states[:,:,3] = np.array([[0,0,0,0],
                                      [0,1,0,0],
                                      [0,0,1,0],
                                      [1,0,0,1]]) 
    # Note : this verison of the tmaze task implemented in DEM_demo_MDP_XX.m,
    # is different from the usual T-maze task in which the task ends when the agent reaches the goal
    # Here, the agent accumulates the reward / aversive stimulus  across 2 timesteps if it 
    # doesnt visit the hint 
    
    #2. Transition between context states --> The agent cannot act so there is only one :
    B_context_states = np.zeros((2,2,1))
    B_context_states[...,0] = np.eye(2)
    
    b = [200*B_behav_states,200*B_context_states]
    
    
    c_extero = np.zeros((5,T))
    
    c_intero = np.array([
        [0,0 ,0   ],      # Neutral outcome
        [0,rs,rs  ],  # We prefer winning asap
        [0,la,la  ]     # losing is always less preferable
    ])
    c = [c_extero,c_intero]
    # print("Done!")
    
    
    U = np.array([
        [0,0],
        [1,0],
        [2,0],
        [3,0]
    ]).astype(int)
    
    #Habits
    e = np.ones((U.shape[0],))

    return a,b,c,d,e,U


def get_jax_T_maze_model(pHa,pWin,initial_hint_confidence,la,rs,context_belief):
    """
    pinit : prob of reward initial position being left / right
    pHA : probability of clue giving the correct index position
    pWin : probability of winning if we are in the correct position
    """
    # print("T-maze gen. model set-up ...  ",end='')
    T = 3

    k_conf = 2.0
    d = [jnp.array([128.0,1.0,1.0,1.0]),jnp.array([context_belief*k_conf,(1.0-context_belief)*k_conf])]

        #Mapping from states to observed positions & hints
    # A_extero = np.zeros(((5,4,2)))
    v_0 = 200*1.0 # value for known cells
    q_0 = initial_hint_confidence*pHa   # prior belief about clue epistemic value
    q_1 = initial_hint_confidence*(1-pHa)
    A_extero_right= jnp.array([
        [v_0,0.0,0.0,0.0], # Position start
        [0.0,v_0,0.0,0.0], # Position right
        [0.0,0.0,v_0,0.0], # Position left
        [0.0,0.0,0.0,q_0], # Cue pointing right
        [0.0,0.0,0.0,q_1]  # Cue pointing left
    ])
    A_extero_left = jnp.array([
        [v_0,0.0,0.0,0.0], # Position start
        [0.0,v_0,0.0,0.0], # Position right
        [0.0,0.0,v_0,0.0], # Position left
        [0.0,0.0,0.0,q_1], # Cue pointing right
        [0.0,0.0,0.0,q_0]  # Cue pointing left
    ])
    A_extero = jnp.stack([A_extero_right,A_extero_left],axis=-1)
    # print(A_extero.shape)
    # return
    # Interoceptive (reward) signals :
    # A_intero = np.zeros((3,4,2))
    A_intero_right = jnp.array([
        [1.0,0.0   ,0.0   ,1.0],     # Neutral
        [0.0,pWin  ,1-pWin,0.0], # Win
        [0.0,1-pWin,pWin  ,0.0]  # Lose
    ])*200.0
    
    A_intero_left = jnp.array([
        [1.0,0.0,0.0,1.0],     # Neutral
        [0.0,1-pWin,pWin,0.0], # Win
        [0.0,pWin,1-pWin,0.0]  # Lose
    ])*200.0
    A_intero = jnp.stack([A_intero_right,A_intero_left],axis=-1)   
    a = [A_extero,A_intero]
    
    
    
    #1. Transition between behavioural states --> 4 actions
    # - 0 --> Move to start from any state, or stay in the left / right cells if already there
    B_behav_states_start = jnp.array([[1,0,0,1],
                                      [0,1,0,0],
                                      [0,0,1,0],
                                      [0,0,0,0]])
    # - 1 --> Move to choose right from start or hint, else stay where you are
    B_behav_states_right = jnp.array([[0,0,0,0],
                                      [1,1,0,1],
                                      [0,0,1,0],
                                      [0,0,0,0]])
    # - 2 --> Move to choose left from start or hint, else stay where you are
    B_behav_states_left = jnp.array([[0,0,0,0],
                                      [0,1,0,0],
                                      [1,0,1,1],
                                      [0,0,0,0]])  
    # - 3 --> Move to hint from start, else stay where you are
    B_behav_states_clue = jnp.array([[0,0,0,0],
                                      [0,1,0,0],
                                      [0,0,1,0],
                                      [1,0,0,1]]) 
    B_behav_states = jnp.stack([B_behav_states_start,B_behav_states_right,B_behav_states_left,B_behav_states_clue],axis=-1)
    # Note : this verison of the tmaze task implemented in DEM_demo_MDP_XX.m,
    # is different from the usual T-maze task in which the task ends when the agent reaches the goal
    # Here, the agent accumulates the reward / aversive stimulus  across 2 timesteps if it 
    # doesnt visit the hint 

    #2. Transition between context states --> The agent cannot act so there is only one :
    B_context_states = jnp.expand_dims(jnp.eye(2),-1)
    
    b = [200*B_behav_states,200*B_context_states]
    
    
    c_extero = jnp.zeros((5,T))
    
    c_intero = jnp.array([
        [0,0 ,0   ],      # Neutral outcome
        [0,rs,rs  ],  # We prefer winning asap
        [0,la,la  ]     # losing is always less preferable
    ])
    c = [c_extero,c_intero]
    # print("Done!")
    
    U = jnp.array([
        [0,0],
        [1,0],
        [2,0],
        [3,0]
    ]).astype(int)
    
    e = jnp.ones((U.shape[0],))
    
    return a,b,c,d,e,U



if __name__ =="__main__":
    
    pHa  = 0.5
    pWin = 0.98
    initial_hint_confidence = 200
    la = -4
    rs = 2
    context_belief = 0.5
    a,b,c,d,e,U = get_jax_T_maze_model(pHa,pWin,initial_hint_confidence,la,rs,context_belief)
    
    print(a)
    print(b)
    print(c)
    print(d)