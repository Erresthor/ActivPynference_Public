import numpy as np

from actynf.base.function_toolbox import normalize
from utils import sub2ind,ind2sub,distance,discretized_distribution_from_value


def behavioural_process(grid_size,start_idx,end_idx,n_feedback_ticks):

    flattened_grid = np.zeros(grid_size).flatten()
    Ns = flattened_grid.shape[0]

    # Starting position
    d0 = np.zeros((Ns,))
    if (type(start_idx)==list):
        start_pos = [sub2ind(grid_size,ix) for ix in start_idx]
        for pos in start_pos : 
            d0[pos] = 1.0
    else:
        start_pos = sub2ind(grid_size,start_idx)
        d0[start_pos] = 1.0
    d = [normalize(d0)]

    # Feedback values : 
    a0 = np.zeros((n_feedback_ticks,Ns))
    for idx,state in enumerate(flattened_grid):
        cellx,celly = ind2sub(grid_size,idx)
        # cell_dist = np.ravel_multi_index()
        distance_to_goal = distance([cellx,celly],end_idx,True,grid_size[0])
        a_val = discretized_distribution_from_value(1.0-distance_to_goal,n_feedback_ticks)
        a0[:,idx] = a_val
    a = [a0]

    # Transition matrices
    # A lot of possible actions, we discretize them as follow : 
    # 9 possible angles x 9 possible mean positions x 3 possible distances = 243 possible actions
    # To sample efficiently, we assume that subjects  entertain different hypotheses regarding what actions
    # affect the feedback, that they sample simultaneously
    
    # Angle mappings : 
    angle_maps = [[0,0],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0], [-1,1]]
    #angle degrees :NA,   0 ,   45,   90,   135,  180,    225,  270,   315
    
    # Warning ! VERTICAL axis (x coords) is inverted for numpy arrays !
    B_angle = np.zeros((Ns,Ns,9))
    for from_state in range(Ns):
        from_x,from_y = ind2sub(grid_size,from_state)
        for action in range(B_angle.shape[-1]):
            to_x = from_x - angle_maps[action][0] # Inversion : going up is lowering your line value
            to_y = from_y + angle_maps[action][1]  
            if ((to_x<0) or (to_x>=grid_size[0]) or (to_y<0) or (to_y>=grid_size[1])) :
                to_state = from_state
            else :
                to_state = sub2ind(grid_size,(to_x,to_y))
            B_angle[to_state,from_state,action]= 1.0

    # All the other action modalities are neutral w.r.t the hidden states
    # i.e. their state to state mapping is an identity matrix
    B_mean_pos = np.zeros((Ns,Ns,9))
    for action in range(B_mean_pos.shape[-1]):
        B_mean_pos[:,:,action] = np.eye(Ns)
    
    B_distances = np.zeros((Ns,Ns,3))
    for action in range(B_distances.shape[-1]):
        B_distances[:,:,action] = np.eye(Ns)

    # To simplify, let's assume that only the angle input is connected to the process :
    b = [B_angle]
    u = np.array(range(b[0].shape[-1])) # allowable actions

    # We receive action outputs from 3 differents sources
    c = [np.linspace(0,n_feedback_ticks-1,n_feedback_ticks)]

    e = np.ones(u.shape)
    # maze_process = mdp_layer("beh_process","process",a,b,c,d,e,u,T,Th,in_seed=seed)
    return a,b,c,d,e,u

def naive_model(parameters,action_model="angle"):
    '''
    A function defining a generic neurofeedback model depending on a few criteria:
    cognitive_layout : a list of the cognitive dimensions we wish to model
    feedback_Nticks : how we wish to discretize the feedback
    '''
    # MAIN ASSUMPTION : HIDDEN STATES ARE DIRECTLY OBSERVABLE USING THE FEEDBACK
    n_feedback_ticks = parameters["N_feedback_ticks"]
    Ns = n_feedback_ticks
    initial_action_mapping_confidence = parameters["b_str_init"]


    d0 = normalize(np.ones((n_feedback_ticks,))) 
        # Uninformed prior on hidden state position before the task starts
    d = [d0]

    # The feedback is a one-dimensionnal information related to the cognitive dimensions
    feedback = np.zeros((n_feedback_ticks,Ns))
    
    for k in range(n_feedback_ticks):
        feedback[k,k] = 1.0 # Assume that the feedback seen is directly related to the hidden state
                            # (no hidden state)
        # Default : dimension 0 !
    a = [feedback]

    # Action modalities : there are various ways of modeling those actions. 
    # Here, we consider that the subject may entertain 3 different models : 
    # 1. A model where the angle drives the feedback  (9 actions)
    # 2. A model where the position of the point drives the feedback (9 actions)
    # 3. A model where the distance between points drives the feedback (3 actions)
    if (action_model=="angle"):
        n_possible_actions = 9
    elif(action_model=="position"):
        n_possible_actions = 9
    elif(action_model=="distance"):
        n_possible_actions = 3
    
    # An initially naive model !
    b0 = normalize(np.ones((Ns,Ns,n_possible_actions)))*initial_action_mapping_confidence
    b = [b0]

    # Assume a linear preference matrix c = ln p(o)
    c = [np.linspace(0,n_feedback_ticks-1,n_feedback_ticks)]
    # # Non linear preference matrix
    # c0 = [1.0]
    # for x in range(1,number_of_ticks):
    #     c0.append(c0[-1]*2)
    # c = [np.array(c0)]

    u = np.array(range(b[0].shape[-1]))
    e = np.ones(u.shape)

    return a,b,c,d,e,u

def basic_latent_model(parameters,action_model="angle"):
    '''
    A function defining a generic neurofeedback model depending on a few criteria:
    cognitive_layout : a list of the cognitive dimensions we wish to model
    feedback_Nticks : how we wish to discretize the feedback
    '''
    
    # MAIN ASSUMPTION : HIDDEN STATES ARE *NOT* DIRECTLY OBSERVABLE USING THE FEEDBACK
    # BUT THE FEEDBACK GIVES AN ESTIMATE OF THE CURRENT LATENT STATE VALUE
    n_feedback_ticks = parameters["N_feedback_ticks"]
    Ns = parameters["Ns_latent"]
    initial_action_mapping_confidence = parameters["b_str_init"]
    
    

    d0 = normalize(np.ones((Ns,))) 
        # Uninformed prior on hidden state position before the task starts
    d = [d0]

    # The feedback is a one-dimensionnal information related to the latent state
    a0 = np.zeros((n_feedback_ticks,Ns))
    for idx,state in enumerate(range(Ns)):
        distance_to_goal = 1.0 - (state/(Ns-1.0))
        a_val = discretized_distribution_from_value(1.0-distance_to_goal,n_feedback_ticks)
        a0[:,idx] = a_val
    a = [a0]
    
    # Action modalities : there are various ways of modeling those actions. 
    # Here, we consider that the subject may entertain 3 different models : 
    # 1. A model where the angle drives the feedback  (9 actions)
    # 2. A model where the position of the point drives the feedback (9 actions)
    # 3. A model where the distance between points drives the feedback (3 actions)
    if (action_model=="angle"):
        n_possible_actions = 9
    elif(action_model=="position"):
        n_possible_actions = 9
    elif(action_model=="distance"):
        n_possible_actions = 3
    
    # An initially naive model !
    b0 = normalize(np.ones((Ns,Ns,n_possible_actions)))*initial_action_mapping_confidence
    b = [b0]

    # Assume a linear preference matrix c = ln p(o)
    c = [np.linspace(0,n_feedback_ticks-1,n_feedback_ticks)]

    # # Non linear preference matrix
    # c0 = [1.0]
    # for x in range(1,number_of_ticks):
    #     c0.append(c0[-1]*2)
    # c = [np.array(c0)]

    u = np.array(range(b[0].shape[-1]))
    e = np.ones(u.shape)
    
    return a,b,c,d,e,u

def grid_latent_model(parameters,action_model="angle"):
    '''
    A function defining a generic neurofeedback model depending on a few criteria:
    cognitive_layout : a list of the cognitive dimensions we wish to model
    feedback_Nticks : how we wish to discretize the feedback
    '''
    # MAIN ASSUMPTION : THERE ARE 4 HIDDEN STATES !
    # 1. x current position
    # 2. y current position
    # 3. x goal
    # 4. y goal
    n_feedback_ticks = parameters["N_feedback_ticks"]
    grid_size = parameters["grid_size"]
    initial_action_mapping_confidence = parameters["b_str_init"]
    
    
    Ns = [grid_size[0],grid_size[1],grid_size[0],grid_size[1]]

    # Uninformed prior on hidden state position before the task starts
    # We start oblivious to the starting state!
    d = [normalize(np.ones((s,))) for s in Ns ]
    
    # ASSUME INITIAL ORIENTATION (TO AVOID SYMMETRY-INDUCED AMBIGUITY)
    d[0][:int(d[0].shape[0]/2)] += 1.0
    d[1][int(d[1].shape[0]/2):] += 1.0
    # d[2][0] += 200
    # d[3][6] += 200
    
    
    # Here, the subject assumes the feedback
    # is a one-dimensionnal information related to
    # the distance between my position and the goal !
    feedback_raw_vals =  np.zeros(tuple(Ns))
    feedback = np.zeros((n_feedback_ticks,) + tuple(Ns))
    for x in range(Ns[0]):
        for y in range(Ns[1]):
            for xgoal in range(Ns[2]):
                for ygoal in range(Ns[3]):
                    expected_linear_feedback = 1.0-distance((x,y),(xgoal,ygoal),True,7)
                    # print(expected_linear_feedback)
                    
                    feedback_dist = discretized_distribution_from_value(expected_linear_feedback,n_feedback_ticks)
                    feedback[:,x,y,xgoal,ygoal] = feedback_dist
                    feedback_raw_vals[x,y,xgoal,ygoal] = expected_linear_feedback
            
    a = [np.ones(feedback.shape) + 200*feedback]

    # # Show what level of feedback we get depending if the goal state is at (0,6)
    # img = np.zeros((Ns[0],Ns[1]))
    # img = expected_linear_feedback = 1.0-distance((x,y),(0,6),True,7)
    # plt.imshow(img)
    # plt.show()

    # Action modalities : there are various ways of modeling those actions. 
    # Here, we consider that the subject may entertain 3 different models : 
    # 1. A model where the angle drives the feedback  (9 actions)
    # 2. A model where the position of the point drives the feedback (9 actions)
    # 3. A model where the distance between points drives the feedback (3 actions)
    if (action_model=="angle"):
        n_possible_actions = 9
    elif(action_model=="position"):
        n_possible_actions = 9
    elif(action_model=="distance"):
        n_possible_actions = 3
    
    # An initially naive model !
    # effect of actions on x:
    b0 = normalize(np.ones((Ns[0],Ns[0],n_possible_actions)))*initial_action_mapping_confidence
    b1 = normalize(np.ones((Ns[1],Ns[1],n_possible_actions)))*initial_action_mapping_confidence
    # assume actions have no effect on the goal
    b2 = np.expand_dims(np.eye(Ns[2]),-1)
    b3 = np.expand_dims(np.eye(Ns[3]),-1)
    b = [b0,b1,b2,b3]


    # Assume a linear preference matrix c = ln p(o)
    c = [np.linspace(0,n_feedback_ticks-1,n_feedback_ticks)]
    # Non linear preference matrix
    # c0 = [1.0]
    # for x in range(1,number_of_feedback_ticks):
    #     c0.append(c0[-1]*2)
    # c = [np.array(c0)]

    u = np.zeros((n_possible_actions,len(Ns)))
    for act in range(n_possible_actions):
        u[act,:] = np.array([act,act,0,0])
    u = u.astype(int)
    e = np.ones(u.shape[0])
    return a,b,c,d,e,u





