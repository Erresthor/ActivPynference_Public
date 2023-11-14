import numpy as np, matplotlib.pyplot as plt, sys,os

from actynf.base.function_toolbox import normalize
from actynf.layer.model_layer import mdp_layer
from actynf.layer.layer_link import establish_layerLink
from actynf.architecture.network import network

number_of_ticks = 5

GRID_SIZE = 7


def distance(tuple1,tuple2,normed=False,grid_size=2):
    linear_dist =  np.sqrt((tuple1[0]-tuple2[0])+(tuple1[1]-tuple2[1]))
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
        sx = x*number_of_ticks
        int_sx = int(sx)  # The lower index
        float_sx = sx-int_sx  # How far this lower index is from the true value
        return_distribution[int_sx] = 1.0-float_sx  # The closer to the true value, the higher the density
        return_distribution[int_sx+1] = float_sx
    return return_distribution

# Python version of matlab functions :
def sub2ind(array_shape, sub_tuple):
    rows, cols = sub_tuple[0],sub_tuple[1]
    if ((rows < 0)or(rows>=array_shape[0])) or ((cols < 0)or(cols>=array_shape[1])) :
        raise ValueError(str(sub_tuple) + " is outside the range for array shape " + str(array_shape))
    return cols*array_shape[0] + rows

def ind2sub(array_shape, ind):
    rows = (ind // array_shape[1])
    cols = (ind % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return rows, cols

def build_maze(grid_size,start_idx,end_idx,
        n_feedback_ticks,
        maze_array,
        T,Th,seed=None):

    
    flattened_grid = np.zeros(grid_size).flatten('F')
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
    for state,idx in enumerate(flattened_grid):
        cell_dist = ind2sub(grid_size,idx)
        distance_to_goal = distance(cell_dist,end_idx,True,grid_size[0])

        a_val = discretized_distribution_from_value(distance_to_goal,n_feedback_ticks)
        a0[:,idx] = a_val
    a = [a0]

    # Transition matrices
    # A lot of possible actions, we discretize them as follow : 
    # 9 possible angles x 9 possible mean positions x 3 possible distances = 243 possible actions
    # Angle mappings : 
    angle_maps = [[0,0],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0], [-1,1]]
    #angle degrees :NA,   0 ,   45,   90,   135,  180,    225,  270,   315
    B_angle = np.zeros((Ns,Ns,9))
    for from_state in range(Ns):
        from_x,from_y = ind2sub(grid_size,from_state)
        for action in range(angle_maps):
            to_x = from_x + angle_maps[action][0]
            to_y = from_y + angle_maps[action][1]
            if ((to_x<0) or (to_x>grid_size[0]) or (to_y<0) or (to_y>grid_size[1])) :
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



    u   = np.array([[ 1, 0],
                    [-1, 0],
                    [ 0, 1],
                    [ 0,-1],
                    [ 0, 0]]) # allowable actions
    Nu = u.shape[0]

    B = np.zeros((Ns,Ns,Nu))
    for from_x in range(maze_array.shape[0]):
        for from_y in range(maze_array.shape[1]):
            s = sub2ind(maze_array.shape,(from_x,from_y))
            for u_ix in range(Nu):
                try :
                    ss = sub2ind(maze_array.shape,(from_x + u[u_ix,0] ,from_y + + u[u_ix,1]))
                    B[ss,s,u_ix] = 1
                except:
                    B[s,s,u_ix] = 1
    b = [B]

    c1 = np.array([2,-2])
    c2 = np.zeros((Ns,))
    Xtarget,Ytarget = end_idx[0],end_idx[1]
    Xtarget,Ytarget = end_idx[1],end_idx[0]
    for c_ix in range(Ns):
        x,y = ind2sub(maze_array.shape,c_ix)
        c2[c_ix] = -1.0*np.sqrt((Xtarget-x)*(Xtarget-x)+(Ytarget-y)*(Ytarget-y))# - 1.0
    #c2[sub2ind(maze_array.shape,(Ytarget,Xtarget))] = 0
    c = [c1,c2]

    U = np.array(range(Nu))
    e = np.ones(U.shape)
    maze_process = mdp_layer("maze_environment","process",a,b,c,d,e,U,T,Th,in_seed=seed)
    return maze_process

def model(number_of_ticks,action_model="angle",
          initial_action_mapping_confidence = 0.1):
    '''
    A function defining a generic neurofeedback model depending on a few criteria:
    cognitive_layout : a list of the cognitive dimensions we wish to model
    feedback_Nticks : how we wish to discretize the feedback
    '''

    d0 = normalize(np.ones((number_of_ticks,))) # Uninformed prior on feedback before the task starts
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
        b0 = np.zeros((Ns,Ns,9))
        # If true mapping :
        b0[:,:,0] = np.eye(Ns)  # Angle 0 = no angle
        b0[:,:,1] = np.eye(Ns)  # Angle 1 = around 0°
    elif(action_model=="position"):
        b0 = np.zeros((Ns,Ns,9))
    elif(action_model=="distance"):
        b0 = np.zeros((Ns,Ns,3))

    # If naive model :
    b0 = np.ones(b0.shape)*initial_action_mapping_confidence
     


    # Linear preference matrix c = ln p(o)
    c = [np.linspace(0,number_of_ticks-1,number_of_ticks)]

    e = None
    u = None
    return a,b,c,d,e,u