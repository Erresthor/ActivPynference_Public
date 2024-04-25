import numpy as np



def build_tmaze_process(pinit,pHA,pWin):
    """
    pinit : prob of reward initial position being left / right
    pHA : probability of clue giving the correct index position
    pWin : probability of winning if we are in the correct position

    This function returns a mdp_layer representing the t-maze environment.
    """
    print("T-maze gen. process set-up ...  ",end='')

    T = 3  # The trials are made of 3 timesteps (starting step + 2 others)

    # Initial situation
    d = [np.array([pinit,1-pinit])    ,np.array([1,0,0,0])]
    #  on which side is the cheese | where is the mouse 
    Ns = [arr.shape[0] for arr in d] # Number of states
    
    # Transition matrixes between hidden states=
    # a. Transition between cheese states --> the cheese doesn't move during the trial, and the mouse can't make it move :
    B_context_states = np.array([[[1],[0]],
                                 [[0],[1]]])
    # b. Transition between mouse position states --> 4 actions possible for the mouse
    B_behav_states = np.zeros((Ns[1],Ns[1],Ns[1]))

    # - 0 --> Move to start from any state
    B_behav_states[0,:,0] = 1          
    # - 1 --> Move to clue from start, else go to start
    B_behav_states[:,:,1] = np.array([[0,1,1,1],
                                      [1,0,0,0],
                                      [0,0,0,0],
                                      [0,0,0,0]])
    # - 2 --> Move to choose left from start or hint, else go to start
    B_behav_states[:,:,2] = np.array([[0,0,1,1],
                                      [0,0,0,0],
                                      [1,1,0,0],
                                      [0,0,0,0]])  
    
    # - 3 --> Move to choose right from start or hint, else go to start
    B_behav_states[:,:,3] = np.array([[0,0,1,1],
                                      [0,0,0,0],
                                      [0,0,0,0],
                                      [1,1,0,0]])
    b = [B_context_states, B_behav_states]
    # Note : as you can see, the mouse can't go to right then left or left then right : every trial, it has to make a decision between the two.

    # Active Inference also revolves around a state-observation correspondance that we describe here :
    

    # 1. Mapping from states to observed hints, depending on cheese & mouse states
    #
    # [ .  . ]  No hint
    # [ .  . ]  Left Hint            Rows = observations
    # [ .  . ]  Right Hint
    # Left Right
    # Columns = cheese state
    A_obs_hints = np.zeros((3,Ns[0],Ns[1]))
    A_obs_hints[0,:,:] = 1
    A_obs_hints[:,:,1] = np.array([[0,0],
                             [pHA, 1-pHA],
                             [1-pHA,pHA]]) # We only get the clue if the mouse moves to state 1
    
    # 2. Mapping from states to outcome (win / loss / null), depending on cheese & mouse states
    #
    # [ .  . ]  Null
    # [ .  . ]  Win           Rows = observations
    # [ .  . ]  Loss
    # Left Right
    # Columns = cheese state
    A_obs_outcome = np.zeros((3,Ns[0],Ns[1]))
    A_obs_outcome[0,:,:2] = 1
    A_obs_outcome[:,:,2] = np.array([[0,0],   # If we choose left, what is the probability of achieving win / loss 
                             [pWin, 1-pWin],
                             [1-pWin,pWin]]) # Choice gives an observable outcome
                   # If true = left, right
    A_obs_outcome[:,:,3] = np.array([[0,0],     # If we choose right, what is the probability of achieving win / loss 
                                     [1-pWin, pWin],
                                     [pWin,1-pWin]]) # Choice gives an observable outcome
                  # If true = left, right
    
    # 3. Mapping from mouse position states to observed mouse position
    #
    # [ .  .  .  .] start
    # [ .  .  .  .] hint
    # [ .  .  .  .] choose left         Row = Behaviour state
    # [ .  .  .  .] choose right
    #  s   h  l  r
    #
    # 3rd dimension = observed behaviour
    # The 2nd dimension maps the dependance on cheese state (unvariant)
    A_obs_behaviour = np.zeros((Ns[1],Ns[0],Ns[1]))
    for i in range (Ns[1]) :
        A_obs_behaviour[i,:,i] = np.array([1,1])
    a = [A_obs_hints,A_obs_outcome,A_obs_behaviour]

    No = [ai.shape[0] for ai in a] # Number of outcomes

    # Finally, we set up the preferences of the environment (this is an environment, thus this is empty) ...
    c = [np.zeros((No[0],T)),np.zeros((No[1],T)),np.zeros((No[2],T))]
    # ... as well as the allowable transitions the mouse can choose :
    u = np.array([[0,0],[0,1],[0,2],[0,3]]).astype(int)
    
    # Habits
    e = np.ones((u.shape[0],))

    # The environment has been well defined and we may now build a mdp_layer using the following constructor : 
    # layer = mdp_layer("T-maze_environment","process",a,b,c,d,e,u,T)
    #     mdp_layer(name of the layer,process or model, a,b,c,d,e,u,T)
    print("Done.")
    return a,b,c,d,e,u




def build_mouse_model(process_b,process_a,process_u,
                      la,rs,T_horizon,
                      initial_clue_confidence = 0.1,action_selection_temperature = 32,mem_loss=0.0):
    """
    true_process_layer : the mdp_layer object where the tmaze environment has been defined
    la : how much the mouse is afraid of adverse outcomes (>0)
    rs : how much the mouse wants to observe cheese (>0)
    T_horizon : how much into the future the mouse will plan before picking its next action
    initial_clue_confidence : how much the mouse knows about the clue reliability
    """
    print("T-maze gen. model set-up ...  ",end='')
    T = 3

    #  The mouse knows where it stands in the maze initially, but it doesn't know where the cheese will spawn : this is something that
    # it will need to learn !
    d = [np.array([0.25,0.25]),np.array([1,0,0,0])]

    
    # Transition matrixes between hidden states ( = control states)
    b=[]
    for b_fac_proc in (process_b):
        b.append(np.copy(b_fac_proc)*200)
    b[1]=b[1]+10
    # The mouse knows how its action will affect the general situation. The mouse does not need
    # to learn that element . Be aware that too much uncertainty in some situations may prove hard to resolve for our
    # artifical subjects.


    a = []
    for a_mod_proc in (process_a):
        a.append(np.copy(a_mod_proc)*200)
    a[0][:,:,1] = initial_clue_confidence*np.array([[0,0],
                                                    [0.25,0.25],
                                                    [0.25,0.25]])  
    # The mouse already knows how the cheese position and its own position in the 
    # maze relates relates to its probability to observe cheese. It also knows where
    # it is in the maze at all times. It knows this because it knows where it isn't ;)
    # However, the mouse still has to learn the reliability of the clue.


    # Finally, the preferences of the mouse are governed by the experimenter through the rs/la weights.
    No = [ai.shape[0] for ai in a]

    C_hints = np.zeros((No[0],T))
    C_win_loss = np.zeros((No[1],T))
    C_win_loss = np.array([[0,0,0],     #null
                           [0,rs,rs/2.0],  #win : as you can see, the mouse would much rather find the cheese at timestep 2 rather than 3. Feel free to play with this factor.
                           [0,-la,-la]]) #loss
    C_observed_behaviour = np.zeros((No[2],T))
    c = [C_hints,C_win_loss,C_observed_behaviour]
    # The mouse has no preference towards seeing a clue or being in a given position. However, it does have a preference regarding
    # the outcome of the trial (i.e. seeing the cheese or the mousetrap)
    
    # The allowable actions have been defined earlier
    u = process_u
    # u = np.array([[0,0],[0,1],[0,2],[0,3]]).astype(int)
    
    # Habits
    e = np.ones((u.shape[0],))

    return a,b,c,d,e,u