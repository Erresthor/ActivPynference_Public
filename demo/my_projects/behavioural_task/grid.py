import numpy as np, matplotlib.pyplot as plt, sys,os

from actynf.base.function_toolbox import normalize
from actynf.layer.model_layer import mdp_layer
from actynf.architecture.network import network

from actynf import NO_STRUCTURE

def distance(tuple1,tuple2,normed=False,grid_size=2):
    linear_dist =  np.sqrt((tuple1[0]-tuple2[0])*(tuple1[0]-tuple2[0])+(tuple1[1]-tuple2[1])*(tuple1[1]-tuple2[1]))
    if normed :
        assert grid_size>1,"Grid should be bigger"
        gs = grid_size-1
        return linear_dist/np.sqrt(gs*gs+gs*gs)
    return linear_dist

def discretized_distribution_from_value(x,number_of_ticks):
    assert number_of_ticks>1,"There should be at least 2 different distribution values"
    return_distribution = np.zeros((number_of_ticks,))
    if (x<0.0):
        return_distribution[0] = 1.0 
    elif (x>=1.0):
        return_distribution[-1] = 1.0
    else :
        sx = x*(number_of_ticks-1)
        int_sx = int(sx)  # The lower index
        float_sx = sx-int_sx  # How far this lower index is from the true value
        return_distribution[int_sx] = 1.0-float_sx  # The closer to the true value, the higher the density
        return_distribution[int_sx+1] = float_sx
    return return_distribution

def mat_sub2ind(array_shape, sub_tuple):
    rows, cols = sub_tuple[0],sub_tuple[1]
    if ((rows < 0)or(rows>=array_shape[0])) or ((cols < 0)or(cols>=array_shape[1])) :
        raise ValueError(str(sub_tuple) + " is outside the range for array shape " + str(array_shape))
    return cols*array_shape[0] + rows

def mat_ind2sub(array_shape, ind):
    rows = (ind // array_shape[1])
    cols = (ind % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return rows, cols


# Python version
def sub2ind(array_shape, sub_tuple):
    """ For integers only !"""
    rows, cols = sub_tuple[0],sub_tuple[1]
    return rows*array_shape[1] + cols

def ind2sub(array_shape, ind):
    """ For integers only !"""
    rows = ind // array_shape[1]
    cols = ind % array_shape[1]
    return rows, cols

def cartesian(arrays, out=None):
    """
    Generate a Cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the Cartesian product of.
    out : ndarray
        Array to place the Cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing Cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

def behavioural_process(grid_size,start_idx,end_idx,
        n_feedback_ticks,
        T,Th,seed=None):

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

    # All the other action modalities are neutral i.r.t the hidden states
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
    Nu = u.shape[0]

    # We receive action outputs from 3 differents sources
    c = [np.linspace(0,n_feedback_ticks-1,n_feedback_ticks)]

    e = np.ones(u.shape)
    maze_process = mdp_layer("beh_process","process",a,b,c,d,e,u,T,Th,in_seed=seed)
    return maze_process

def basic_model(number_of_ticks,T,Th,
        action_model="angle",
        initial_action_mapping_confidence = 0.1,
        structure_hypothesis = NO_STRUCTURE,
        state_cap = None,action_cap = None,
        seed=None):
    '''
    A function defining a generic neurofeedback model depending on a few criteria:
    cognitive_layout : a list of the cognitive dimensions we wish to model
    feedback_Nticks : how we wish to discretize the feedback
    '''
    # MAIN ASSUMPTION : HIDDEN STATES ARE DIRECTLY OBSERVABLE USING THE FEEDBACK
    Ns = number_of_ticks


    d0 = normalize(np.ones((number_of_ticks,))) 
        # Uninformed prior on hidden state position before the task starts
    d = [d0]

    # The feedback is a one-dimensionnal information related to the cognitive dimensions
    feedback = np.zeros((number_of_ticks,Ns))
    for k in range(number_of_ticks):
        feedback[k,k] = 1.0 # Assume that the feedback seen is directly related to the hidden state
                            # (no hidden state)
        # Default : dimension 0 !
    a = [feedback]

    # Action modalities : there are various ways of modeling those actions. 
    # Here, we consider that the subject may entertain 3 different models : 
    # 1. A model where the angle drives the feedback  (9 actions)
    # 2. A model where the position of the point drives the feedback (9 actions)
    # 3. A model where the distance between points drives the feedback (3 actions)
    Ns = number_of_ticks
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
    c = [np.linspace(0,number_of_ticks-1,number_of_ticks)]
    c = [np.linspace(-number_of_ticks,number_of_ticks-1,number_of_ticks)]
    # # Non linear preference matrix
    # c0 = [1.0]
    # for x in range(1,number_of_ticks):
    #     c0.append(c0[-1]*2)
    # c = [np.array(c0)]

    u = np.array(range(b[0].shape[-1]))
    e = np.ones(u.shape)
    maze_model = mdp_layer("subj_"+action_model,"model",a,b,c,d,e,u,T,Th,in_seed=seed)


    maze_model.hyperparams.alpha = 32
    maze_model.hyperparams.cap_action_explo = action_cap
    maze_model.hyperparams.cap_state_explo = state_cap

    maze_model.learn_options.eta = 10
    maze_model.learn_options.learn_a = False
    maze_model.learn_options.learn_b = True
    maze_model.learn_options.learn_d = True
    maze_model.learn_options.assume_state_space_structure = structure_hypothesis
    return maze_model

def naive_grid_model(number_of_ticks,grid_size,T,Th,
        action_model="angle",
        initial_action_mapping_confidence = 0.1,
        structure_hypothesis = NO_STRUCTURE,
        state_cap = None,action_cap = None,
        seed=None):
    '''
    A function defining a generic neurofeedback model depending on a few criteria:
    cognitive_layout : a list of the cognitive dimensions we wish to model
    feedback_Nticks : how we wish to discretize the feedback
    '''
    # MAIN ASSUMPTION : HIDDEN STATES ARE DIRECTLY OBSERVABLE USING THE FEEDBACK
    Ns = number_of_ticks


    d0 = normalize(np.ones((number_of_ticks,))) 
        # Uninformed prior on hidden state position before the task starts
    d = [d0]

    # The feedback is a one-dimensionnal information related to the cognitive dimensions
    feedback = np.zeros((number_of_ticks,Ns))
    for k in range(number_of_ticks):
        feedback[k,k] = 1.0 # Assume that the feedback seen is directly related to the hidden state
                            # (no hidden state)
        # Default : dimension 0 !
    a = [feedback]

    # Action modalities : there are various ways of modeling those actions. 
    # Here, we consider that the subject may entertain 3 different models : 
    # 1. A model where the angle drives the feedback  (9 actions)
    # 2. A model where the position of the point drives the feedback (9 actions)
    # 3. A model where the distance between points drives the feedback (3 actions)
    Ns = number_of_ticks
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
    c = [np.linspace(0,number_of_ticks-1,number_of_ticks)]
    c = [np.linspace(-number_of_ticks,number_of_ticks-1,number_of_ticks)]
    # # Non linear preference matrix
    # c0 = [1.0]
    # for x in range(1,number_of_ticks):
    #     c0.append(c0[-1]*2)
    # c = [np.array(c0)]

    u = np.array(range(b[0].shape[-1]))
    e = np.ones(u.shape)
    maze_model = mdp_layer("subj_"+action_model,"model",a,b,c,d,e,u,T,Th,in_seed=seed)


    maze_model.hyperparams.alpha = 32
    maze_model.hyperparams.cap_action_explo = action_cap
    maze_model.hyperparams.cap_state_explo = state_cap

    maze_model.learn_options.eta = 10
    maze_model.learn_options.learn_a = False
    maze_model.learn_options.learn_b = True
    maze_model.learn_options.learn_d = True
    maze_model.learn_options.assume_state_space_structure = structure_hypothesis
    return maze_model

def complex_grid_model(number_of_feedback_ticks,
        grid_size,
        T,Th,
        action_model="angle",
        initial_action_mapping_confidence = 0.1,
        structure_hypothesis = NO_STRUCTURE,
        state_cap = None,action_cap = None,
        seed=None):
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
    Ns = [grid_size[0],grid_size[1],grid_size[0],grid_size[1]]

    # Uninformed prior on hidden state position before the task starts
    # We start oblivious to the starting state!
    d = [normalize(np.ones((s,))) for s in Ns ]
    
    # ASSUME ORIENTATION
    d[0][:int(d[0].shape[0]/2)] += 1.0
    d[1][int(d[1].shape[0]/2):] += 1.0
    # d[2][0] += 200
    # d[3][6] += 200
    # Here, the subject assumes the feedback
    # is a one-dimensionnal information related to
    # the distance between my position and the goal !
    img = np.zeros((Ns[0],Ns[1]))

    feedback = np.zeros((number_of_feedback_ticks,) + tuple(Ns))
    for x in range(Ns[0]):
        for y in range(Ns[1]):
            for xgoal in range(Ns[2]):
                for ygoal in range(Ns[3]):
                    expected_linear_feedback = 1.0-distance((x,y),(xgoal,ygoal),True,7)
                    # print(expected_linear_feedback)
                    feedback_dist = discretized_distribution_from_value(expected_linear_feedback,number_of_feedback_ticks)
                    feedback[:,x,y,xgoal,ygoal] = feedback_dist
            img[x,y] = expected_linear_feedback = 1.0-distance((x,y),(0,6),True,7)
    a = [np.ones(feedback.shape) + 1000*feedback]

    # Show what level of feedback we get depending on the goal state
    plt.imshow(img)
    plt.show()

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


    # print([B.shape for B in b])
    # input()
    # Assume a linear preference matrix c = ln p(o)
    c = [np.linspace(-number_of_feedback_ticks,number_of_feedback_ticks-1,number_of_feedback_ticks)]
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
    maze_model = mdp_layer("subj_"+action_model,"model",a,b,c,d,e,u,T,Th,in_seed=seed)


    maze_model.hyperparams.alpha = 32
    maze_model.hyperparams.cap_action_explo = action_cap
    maze_model.hyperparams.cap_state_explo = state_cap

    maze_model.learn_options.eta = 10
    maze_model.learn_options.learn_a = False
    maze_model.learn_options.learn_b = True
    maze_model.learn_options.learn_d = True
    maze_model.learn_options.assume_state_space_structure = [structure_hypothesis,structure_hypothesis,NO_STRUCTURE,NO_STRUCTURE]
    return maze_model