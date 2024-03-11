import numpy as np
import math

import actynf
from tools import gaussian_to_categorical,gaussian_from_distance_matrix,clever_running_mean

import itertools

def allowable_actions(Ns,N_up_actions,N_down_actions,N_useless_actions,allowable_actions_based_on="dim"):
    # Hypothesis 1 : choose one dimension only !
    if allowable_actions_based_on=="dim":
        U = [[0 for k in Ns]] 
            # Initialize with neutral on all dimensions
        for f,ns_f in enumerate(Ns):
            if type(N_up_actions==int):
                N_up = N_up_actions
                N_down = N_down_actions
                N_neutral = N_useless_actions
            else : 
                N_up = N_up_actions[f]
                N_down = N_down_actions[f]
                N_neutral = N_useless_actions[f]
            Nu = N_up + N_down + N_neutral  # For this specific state

            for u in range(1,Nu):
                action_factor = [0 for k in Ns]
                action_factor[f] = u
                U.append(action_factor)

    # Hypothesis 2 : open action selection !
    elif allowable_actions_based_on=="all":
        Nus = []
        for f,ns_f in enumerate(Ns):
            if type(N_up_actions==int):
                N_up = N_up_actions
                N_down = N_down_actions
                N_neutral = N_useless_actions
            else : 
                N_up = N_up_actions[f]
                N_down = N_down_actions[f]
                N_neutral = N_useless_actions[f]
            Nu = N_up + N_down + N_neutral  # For this specific state
            Nus.append(Nu)
        U = list(itertools.product(*[range(nu) for nu in Nus]))
    u = np.array(U).astype(int)
    return u 

def transition_weights(Ns,
                   N_up_actions,N_down_actions,N_neutral_actions,
                   decay_probability,transition_probability):
    assert type(N_up_actions)==type(N_down_actions),"Check your N_X_actions type"
    assert type(N_up_actions)==type(N_neutral_actions),"Check your N_X_actions type"

    # We assume that each successive mental state can be achieved through
    # taking a specific action. To perform the task correctly, the subject
    # has to learn a specific state -> action decision rule !
    # Let's assume that for each state factor, there are N_up_actions to increase the related state dimension, N_down_actions to decrease the related state dimension, and N_useless_actions that don't affect the related state dimension
    
    b = []
    U = []
    for f,ns_f in enumerate(Ns):
        if type(N_up_actions==int):
            N_up = N_up_actions
            N_down = N_down_actions
            N_neutral = N_neutral_actions
        else : 
            N_up = N_up_actions[f]
            N_down = N_down_actions[f]
            N_neutral = N_neutral_actions[f]
        Nu = N_up + N_down + N_neutral  # For this specific state
        b_f = np.zeros((ns_f,ns_f,Nu))

        
        resting_state = 0 # What is the resting state for the subject ?
        # Let's initialize all actions to neutral with a cognitive decay
        for action in range(Nu):
            for from_state in range(ns_f):
                if (from_state == resting_state):
                    to_state = resting_state
                    b_f[to_state,from_state,action] = 1.0
                else : 
                    # Increase of decrease ?
                    to_state = int(from_state + math.copysign(1,resting_state-from_state))
                    # print(int(to_state))
                    b_f[to_state,from_state,action] = decay_probability
                    b_f[from_state,from_state,action] = 1.0 - decay_probability

        # Now, the neutral actions are always the first N_up_actions (to free slot 0)
                    
        # Then come the increasing actions            
        for action in range(N_neutral,N_neutral+N_up,1):
            # The first N_up_actions allow us to go from state i to i+1 along the selected dim
            for from_state in range(ns_f):
                b_f[:,from_state,action] = np.zeros((ns_f,)) # Override previously defined mapping
                if from_state == (ns_f - 1) : # If we are already at the top of this cognitive axis : 
                    b_f[from_state,from_state,action] = 1.0
                else:
                    to_state = from_state + 1
                    b_f[to_state,from_state,action] = transition_probability  # probability of transition
                    b_f[from_state,from_state,action] = 1.0 - transition_probability 
                
        # Finally, the decreasing actions are always the last N_down_actions
        for action in range(N_neutral+N_up,N_neutral+N_up+N_down,1):
            # The next N_down_actions allow us to go from state i to i-1 along the only dim
            for from_state in range(ns_f):
                b_f[:,from_state,action] = np.zeros((ns_f,)) # Override previously defined mapping
                if (from_state == 0) : # If we are already at the lowest of this cognitive axis : 
                    b_f[from_state,from_state,action] = 1.0
                else:
                    to_state = from_state - 1
                    b_f[to_state,from_state,action] = transition_probability  # probability of transition
                    b_f[from_state,from_state,action] = 1.0 - transition_probability 

        b.append(b_f)
    return b

def transition_weights_centered(Ns,
                   N_up_actions,N_down_actions,N_neutral_actions,
                   decay_probability,transition_probability,
                   resting_state_per_factor=None):
    assert type(N_up_actions)==type(N_down_actions),"Check your N_X_actions type"
    assert type(N_up_actions)==type(N_neutral_actions),"Check your N_X_actions type"

    # We assume that each successive mental state can be achieved through
    # taking a specific action. To perform the task correctly, the subject
    # has to learn a specific state -> action decision rule !
    # Let's assume that for each state factor, there are N_up_actions to increase the related state dimension, N_down_actions to decrease the related state dimension, and N_useless_actions that don't affect the related state dimension
    
    b = []
    U = []
    for f,ns_f in enumerate(Ns):
        if type(N_up_actions==int):
            N_up = N_up_actions
            N_down = N_down_actions
            N_neutral = N_neutral_actions
        else : 
            N_up = N_up_actions[f]
            N_down = N_down_actions[f]
            N_neutral = N_neutral_actions[f]
        Nu = N_up + N_down + N_neutral  # For this specific state
        b_f = np.zeros((ns_f,ns_f,Nu))

        try :
            resting_state = resting_state_per_factor[f] # What is the resting state for the subject ?
        except :
            resting_state = 0

        # Let's initialize all actions to neutral with a cognitive decay
        for action in range(Nu):
            for from_state in range(ns_f):
                if (from_state == resting_state):
                    to_state = resting_state
                    b_f[to_state,from_state,action] = 1.0
                else : 
                    # Increase of decrease ?
                    to_state = int(from_state + math.copysign(1,resting_state-from_state))
                    # print(int(to_state))
                    b_f[to_state,from_state,action] = decay_probability
                    b_f[from_state,from_state,action] = 1.0 - decay_probability

        # Now, the neutral actions are always the first N_up_actions (to free slot 0)
                    
        # Then come the increasing actions            
        for action in range(N_neutral,N_neutral+N_up,1):
            # The first N_up_actions allow us to go from state i to i+1 along the selected dim
            for from_state in range(ns_f):
                b_f[:,from_state,action] = np.zeros((ns_f,)) # Override previously defined mapping
                if from_state == (ns_f - 1) : # If we are already at the top of this cognitive axis : 
                    b_f[from_state,from_state,action] = 1.0
                else:
                    to_state = from_state + 1
                    b_f[to_state,from_state,action] = transition_probability  # probability of transition
                    b_f[from_state,from_state,action] = 1.0 - transition_probability 
                
        # Finally, the decreasing actions are always the last N_down_actions
        for action in range(N_neutral+N_up,N_neutral+N_up+N_down,1):
            # The next N_down_actions allow us to go from state i to i-1 along the only dim
            for from_state in range(ns_f):
                b_f[:,from_state,action] = np.zeros((ns_f,)) # Override previously defined mapping
                if (from_state == 0) : # If we are already at the lowest of this cognitive axis : 
                    b_f[from_state,from_state,action] = 1.0
                else:
                    to_state = from_state - 1
                    b_f[to_state,from_state,action] = transition_probability  # probability of transition
                    b_f[from_state,from_state,action] = 1.0 - transition_probability 

        b.append(b_f)
    return b


def transition_prior(Ns,
                   N_up_actions,N_down_actions,N_useless_actions,
                   concentration=1.1,stickiness=1.0):
    assert type(N_up_actions)==type(N_down_actions),"Check your N_X_actions type"
    assert type(N_up_actions)==type(N_useless_actions),"Check your N_X_actions type"

    bprior = []
    for ns_f in Ns:
        if type(N_up_actions==int):
            N_up = N_up_actions
            N_down = N_down_actions
            N_neutral = N_useless_actions
        else : 
            N_up = N_up_actions[f]
            N_down = N_down_actions[f]
            N_neutral = N_useless_actions[f]
        Nu = N_up + N_down + N_neutral  # For this specific state

        sticky_prior_f = np.repeat(np.expand_dims(np.eye(ns_f), axis=-1), Nu, axis=-1)
        full_prior_f = concentration*np.ones((ns_f,ns_f,Nu)) + stickiness*sticky_prior_f
        bprior.append(full_prior_f)
    return bprior



def observation_gaussian_weights(No,Ns,
                        sigma,target_dims=[0],target_state=[-1],
                        method="linear",dist_matrix=None):
    if dist_matrix is None:
        # Assume equivalence of target hidden dimensions for the feedback
        if target_state == None :
            target_state = [-1 for i in target_dims]
        assert len(target_dims) == len(target_state), "Target dimensions and target state in that dimension should be the same length."


        dependent_state_dims = tuple(Ns)
        a0 = np.zeros((No,)+dependent_state_dims)
        dist_array = np.zeros(dependent_state_dims)

        # Itération sur les indices et les valeurs
        for index, value in np.ndenumerate(a0[0,...]):
            reduced_current_coordinates = [(index[i]/(Ns[i]-1.0)) for i in target_dims]
            target_coordinates = [((Ns[k]+i)/(Ns[k]-1.0) if (i<0) else i/(Ns[k]-1.0)) for k,i in enumerate(target_state)]

            if (method=="linear"):
                dist = np.linalg.norm(np.array(reduced_current_coordinates)-np.array(target_coordinates))
                dist_array[index] = dist
            else : 
                raise NotImplementedError("Not implemented method '"+str(method)+"'.")
        
        normalized_distance_matrix = (dist_array/np.max(dist_array))
        return gaussian_from_distance_matrix(No,normalized_distance_matrix,sigma)
    else :
        return gaussian_from_distance_matrix(No,dist_matrix,sigma)

def observation_gaussian_prior(No,Ns,
                        sigma,
                        concentration=1.0,stickiness=1.0,
                        target_dims=[0],target_state=[-1],
                        method="linear",dist_matrix=None):
    return stickiness*observation_gaussian_weights(No,Ns,sigma,target_dims,target_state,method,dist_matrix)+concentration*1.0

def weights_layer_dims(Ns,Nos,
            sigmas, list_of_targets, # For observation matrices
            N_up_actions,N_down_actions,N_useless_actions, # For transition matrix
            observation_concentration = None, observation_stickiness = None,
            decay_probability=None,transition_probability=None, # For transition process
            transition_concentration=None, transition_stickiness=None, # For transition model
            allowable_actions_based_on = "dim",
            mode = "process"):
    """
    Initialize the weights of the layer, namely the active inference usual matrices A,B,D / a,b,c,d,e + U
    This function has two "modes" : process & model. Process returns A,B and D, model returns a,b,c,d,e
    Inputs : 
    - Ns : list of state dimensions of the given layer
    - Nos : list of output dimensions of the given layer
    # DEFINING OBSERVATION MATRICES
    - sigmas : list of observation standard deviations for a given modality. Must be the same length as Nos
    - list_of_targets: a list of lists. Each element of list_of_targets comprises :
        - A list of targeted cognitive dimensions (all between 0 and Ns-1)
        - A list of targeted states in these dimensions (if None, the last state (index = -1) is defaulted)
    # DEFINING TRANSITION MATRICES
    - N_X_actions : the number of actions with a specific effect on a cognitive dimension (go up ? go down ? stay the same ?). 
                    May be a list of integers of length = len(Ns) if we want dimension specific action topography !
    - decay / transition prob : (process only) stochastic parameters on the effect of a specific mental action
    - transition concentration/stickiness : (model only) subject prior beliefs about the transition : b0 = concentration * ones(Ns,Ns,Nu) + stickiness * (eye(Ns) x Nu)
    """
    assert len(Nos)==len(sigmas), "Outcome length mismatch. Check your network parameters. [sigmas]"
    assert len(Nos)==len(list_of_targets), "Outcome length mismatch. Check your network parameters. [targets]"

    a = []
    for No,sigma,target in zip(Nos,sigmas,list_of_targets):
        target_dim = target[0]
        target_state = target[1]
        if mode=="process":
            a.append(observation_gaussian_weights(No,Ns,sigma,target_dim,target_state,method="linear"))
        if mode=="model":
            a.append(observation_gaussian_prior(No,Ns,sigma,observation_concentration,observation_stickiness,target_dim,target_state,method="linear"))

    # Preference matrix : higher observations are prefered !
    MIN_VAL = -10
    MAX_VAL = 10
    c = [np.linspace(MIN_VAL,MAX_VAL,No) for No in Nos]

    if mode=="process":
        b = transition_weights(Ns,N_up_actions,N_down_actions,N_useless_actions,decay_probability,transition_probability)
    elif mode=="model":
        b = transition_prior(Ns,N_up_actions,N_down_actions,N_useless_actions,concentration=transition_concentration,stickiness=transition_stickiness)

    # Fixed starting state :
    if mode=="process":
        d = [np.zeros(ns_f) for ns_f in Ns]
        for df in d : 
            df[0] = 1.0 # Always start trials from the lower states
    elif mode=="model":
        d = [actynf.normalize(np.ones(ns_f)) for ns_f in Ns]

    U = allowable_actions(Ns,N_up_actions,N_down_actions,N_useless_actions,allowable_actions_based_on)

    e = np.ones((U.shape[0],))

    return a,b,c,d,e,U

def weights_layer_dist(Ns,Nos,
            sigmas, dist_matrix, # For observation matrices
            N_up_actions,N_down_actions,N_useless_actions, # For transition matrix
            observation_concentration = None, observation_stickiness = None,
            decay_probability=None,transition_probability=None, # For transition process
            transition_concentration=None, transition_stickiness=None, # For transition model
            allowable_actions_based_on = "dim",
            mode = "process",
            resting_state_per_factor=None):
    """
    Initialize the weights of the layer, namely the active inference usual matrices A,B,D / a,b,c,d,e + U
    This function has two "modes" : process & model. Process returns A,B and D, model returns a,b,c,d,e
    Contrary to the weights_layere function, awaits for a matrix of size prod(Ns) giving the distance to a target state
    instead of a list of targets
    Inputs : 
    - Ns : list of state dimensions of the given layer
    - Nos : list of output dimensions of the given layer
    # DEFINING OBSERVATION MATRICES
    - sigmas : list of observation standard deviations for a given modality. Must be the same length as Nos
    - dist_matrix_(proc/model) : a normalized matrix of distances (between 0 and 1) giving a measure of the distance between
    a target hidden state and any hidden state. Maps the quality of the biomarker.
    # DEFINING TRANSITION MATRICES
    - N_X_actions : the number of actions with a specific effect on a cognitive dimension (go up ? go down ? stay the same ?). 
                    May be a list of integers of length = len(Ns) if we want dimension specific action topography !
    - decay / transition prob : (process only) stochastic parameters on the effect of a specific mental action
    - transition concentration/stickiness : (model only) subject prior beliefs about the transition : b0 = concentration * ones(Ns,Ns,Nu) + stickiness * (eye(Ns) x Nu)
    """
    assert len(Nos)==len(sigmas), "Outcome length mismatch. Check your network parameters. [sigmas]"
    assert len(Nos)==len(dist_matrix), "Outcome length mismatch. Check your network parameters. [dist_matrix]"

    a = []
    for No,sigma,dist_mat in zip(Nos,sigmas,dist_matrix):
        if mode=="process":
            a.append(observation_gaussian_weights(No,Ns,sigma,None,None,dist_matrix=dist_mat,method="linear"))
        if mode=="model":
            a.append(observation_gaussian_prior(No,Ns,sigma,observation_concentration,observation_stickiness,None,None,dist_matrix=dist_mat,method="linear"))

    # Preference matrix : higher observations are prefered !
    MIN_VAL = -10
    MAX_VAL = 10
    c = [np.linspace(MIN_VAL,MAX_VAL,No) for No in Nos]

    if mode=="process":
        b = transition_weights_centered(Ns,N_up_actions,N_down_actions,N_useless_actions,decay_probability,transition_probability,resting_state_per_factor)
    elif mode=="model":
        b = transition_prior(Ns,N_up_actions,N_down_actions,N_useless_actions,concentration=transition_concentration,stickiness=transition_stickiness)

    # Fixed starting state :
    if mode=="process":
        d = [np.zeros(ns_f) for ns_f in Ns]
        for f,df in enumerate(d) : 
            try :
                # If possible, always start trials from the resting state
                df[resting_state_per_factor[f]] = 1.0 
            except:
                # Else, always start trials from the lower states 
                df[0] = 1.0                
    elif mode=="model":
        # Assume uninformed cognitive prior
        d = [actynf.normalize(np.ones(ns_f)) for ns_f in Ns]

    U = allowable_actions(Ns,N_up_actions,N_down_actions,N_useless_actions,allowable_actions_based_on)

    e = np.ones((U.shape[0],))

    return a,b,c,d,e,U

# Define the network
def nf_net_dims(T,Th,
        Ns_proc,Ns_subj,Nos,
        sigmas_proc,list_of_targets_proc,
        sigmas_subj,list_of_targets_subj,observation_concentration,observation_stickiness,
        N_up_actions,N_down_actions,N_neutral_actions,
        p_decay,p_effect,
        transition_concentration,transition_stickiness,
        learning_space_structure=actynf.LINEAR,gen_temp=3.0,
        resting_state_per_factor=None):
    
    A,B,_,D,_,U = weights_layer_dims(Ns_proc,Nos,sigmas_proc,list_of_targets_proc,
                    N_up_actions,N_down_actions,N_neutral_actions,
                    None,None,
                    p_decay,p_effect,
                    None,None,
                    allowable_actions_based_on="dim",mode="process",
                    resting_state_per_factor=resting_state_per_factor)
    
    a,b,c,d,e,_ = weights_layer_dims(Ns_subj,Nos,sigmas_subj,list_of_targets_subj,
                    N_up_actions,N_down_actions,N_neutral_actions,
                    observation_concentration,observation_stickiness,
                    None,None,
                    transition_concentration,transition_stickiness,
                    allowable_actions_based_on="dim",mode="model",
                    resting_state_per_factor=resting_state_per_factor)

    process_layer = actynf.layer("process","process",
                 A,B,c,D,e,U,T)
    
    model_layer = actynf.layer("model","model",
                 a,b,c,d,e,U,T,Th)
    
    model_layer.hyperparams.cap_state_explo = 2
    model_layer.hyperparams.cap_action_explo = 3

    model_layer.learn_options.learn_a = True
    model_layer.learn_options.learn_b = True
    model_layer.learn_options.learn_c = False
    model_layer.learn_options.learn_d = True
    model_layer.learn_options.learn_e = False
    model_layer.learn_options.assume_state_space_structure = learning_space_structure
    model_layer.learn_options.generalize_fadeout_function_temperature = gen_temp
    

    process_layer.inputs.u = actynf.link(model_layer,lambda x : x.u)
    model_layer.inputs.o = actynf.link(process_layer, lambda x : x.o)

    net = actynf.layer_network([process_layer,model_layer],"neurofeedback_training_net")
    return net

def nf_net_dist(T,Th,
        Ns_proc,Ns_subj,Nos,
        sigmas_proc,dist_matrix_proc,
        sigmas_subj,dist_matrix_modl,observation_concentration,observation_stickiness,
        N_up_actions,N_down_actions,N_neutral_actions,
        p_decay,p_effect,
        transition_concentration,transition_stickiness,
        learning_space_structure=actynf.LINEAR,gen_temp=3.0,
        resting_state_per_factor=None):
    
    A,B,_,D,_,U = weights_layer_dist(Ns_proc,Nos,sigmas_proc,dist_matrix_proc,
                    N_up_actions,N_down_actions,N_neutral_actions,
                    None,None,
                    p_decay,p_effect,
                    None,None,
                    allowable_actions_based_on="dim",mode="process",
                    resting_state_per_factor=resting_state_per_factor)
    
    a,b,c,d,e,_ = weights_layer_dist(Ns_subj,Nos,sigmas_subj,dist_matrix_modl,
                    N_up_actions,N_down_actions,N_neutral_actions,
                    observation_concentration,observation_stickiness,
                    None,None,
                    transition_concentration,transition_stickiness,
                    allowable_actions_based_on="dim",mode="model")

    process_layer = actynf.layer("process","process",
                 A,B,c,D,e,U,T)
    
    model_layer = actynf.layer("model","model",
                 a,b,c,d,e,U,T,Th)
    
    model_layer.hyperparams.cap_state_explo = 3
    model_layer.hyperparams.cap_action_explo = 3

    model_layer.learn_options.learn_a = True
    model_layer.learn_options.learn_b = True
    model_layer.learn_options.learn_c = False
    model_layer.learn_options.learn_d = True
    model_layer.learn_options.learn_e = False
    model_layer.learn_options.assume_state_space_structure = learning_space_structure
    model_layer.learn_options.generalize_fadeout_function_temperature = gen_temp
    

    process_layer.inputs.u = actynf.link(model_layer,lambda x : x.u)
    model_layer.inputs.o = actynf.link(process_layer, lambda x : x.o)

    net = actynf.layer_network([process_layer,model_layer],"neurofeedback_training_net")
    return net

def rest_phase_layer(T,Th,
    Ns_proc,Ns_subj,Nos,
    sigmas_proc,dist_matrix_proc,
    sigmas_subj,dist_matrix_modl,observation_concentration,observation_stickiness,
    N_up_actions,N_down_actions,N_neutral_actions,
    p_decay,p_effect,
    transition_concentration,transition_stickiness,
    learning_space_structure=actynf.LINEAR,gen_temp=3.0,
    resting_state_per_factor=None):

    A,B,_,D,_,U = weights_layer_dist(Ns_proc,Nos,sigmas_proc,dist_matrix_proc,
                    N_up_actions,N_down_actions,N_neutral_actions,
                    None,None,
                    p_decay,p_effect,
                    None,None,
                    allowable_actions_based_on="dim",mode="process",
                    resting_state_per_factor=resting_state_per_factor)
    A.append(actynf.normalize(np.ones((1,)+tuple(Ns_proc,))))
        # We had a second observation modality !

    a,b,c,d,e,_ = weights_layer_dist(Ns_subj,Nos,sigmas_subj,dist_matrix_modl,
                    N_up_actions,N_down_actions,N_neutral_actions,
                    observation_concentration,observation_stickiness,
                    None,None,
                    transition_concentration,transition_stickiness,
                    allowable_actions_based_on="dim",mode="model")
    # And we replace the A matrix by one accepting only the "rest" feedback
    # Note : TODO ; update the package to accept optional observations : 
    # - If the observation is in inputs, use it
    # - If not, infer anyways
    a = [actynf.normalize(np.ones((1,)+tuple(Ns_subj,)))]

    c = [np.zeros((1,1))]

    process_layer = actynf.layer("process","process",
                 A,B,c,D,e,U,T)
    
    model_layer = actynf.layer("model","model",
                 a,b,c,d,e,U,T,Th)
    
    model_layer.hyperparams.cap_state_explo = 3
    model_layer.hyperparams.cap_action_explo = 3

    model_layer.learn_options.learn_a = False
    model_layer.learn_options.learn_b = True
    model_layer.learn_options.learn_c = False
    model_layer.learn_options.learn_d = True
    model_layer.learn_options.learn_e = False
    model_layer.learn_options.assume_state_space_structure = learning_space_structure
    model_layer.learn_options.generalize_fadeout_function_temperature = gen_temp
    

    process_layer.inputs.u = actynf.link(model_layer,lambda x : x.u)
    model_layer.inputs.o = actynf.link(process_layer, lambda x : x.o[1])

    net = actynf.layer_network([process_layer,model_layer],"neurofeedback_training_net")
    return net

if __name__ == "__main__":
    T = 15
    Th = 2
    Ns_proc = [5,5,5]  # Intensity of hand movement, right x left x cognitive inhibition
    Nos = [20]
    true_sigmas = [2.0]  # Quite noisy feedback
    belief_sigmas = [0.5]
    targets = [[[0,1,2],[-1,-1,-1]]] # Feedback based on 3 dimensional cognition
    # targets = [[[0],[-1]]]
    N_up = 1
    N_down = 1
    N_neutral = 5
    fullnet = nf_net_dims(T,Th,
                     Ns_proc,Ns_proc,Nos,
                     true_sigmas,belief_sigmas,targets,
                     N_up,N_down,N_neutral,
                     0.1,0.99,1.0,1.0,
                     actynf.LINEAR, 3.0)  # Ability to generalize actions
    
    import os,sys
    import pickle
    savepath = "temp.simu"
    exists = os.path.isfile(savepath)
    if (not(exists)) :
        Nsubj = 1
        Ntrials = 250

        stm_subjs = []
        weight_subjs = []
        for sub in range(Nsubj):
            subj_net = fullnet.copy_network(sub)

            STMs,weights = subj_net.run_N_trials(Ntrials,return_STMs=True,return_weights=True)
            stm_subjs.append(STMs)
            weight_subjs.append(weights)
            
       
        save_this = {
                "stms": stm_subjs,
                "matrices" : weight_subjs
        }
            
        with open(savepath, 'wb') as handle:
            pickle.dump(save_this, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved to :   " + savepath)
        # print(fullnet.layers[0])
    

    # EXTRACT TRAINING CURVES    
    with open(savepath, 'rb') as handle:
        saved_data = pickle.load(handle)
    stms = saved_data["stms"]
    weights = saved_data["matrices"]

    Nsubj = len(stms)
    Ntrials = len(weights[0])-1 # One off because we save the initial weights (= trial 0)

    trial_id = 1

    xs = np.linspace(0,Ntrials,Ntrials)
    ys = np.zeros((Ntrials,))
    tots = np.expand_dims(np.array([0,0,0]),axis=1)
    for trial in range(1,Ntrials+1):
        # print(stms[0][trial][1].o)
        # print(stms[0][trial][0].x)
        # print(np.array([fullnet.layers[0].U[i] for i in stms[0][trial][0].u]))

        xmean = stms[0][trial][0].x
        print(xmean)
        tots = np.concatenate([tots,xmean],axis=1)
        # print(np.round(weights[0][trial][1]["a"][0][:,:,-1,-1],2))
        ys[trial-1] = np.mean(stms[0][trial][0].o)

    import matplotlib.pyplot as plt

    Xs = np.linspace(0,tots.shape[1],tots.shape[1])
    for k in range(tots.shape[0]):
    
        plt.plot(Xs,clever_running_mean(tots[k,:],50))

    # plt.plot(xs,ys)
    plt.show()