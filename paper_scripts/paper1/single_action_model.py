import numpy as np

import actynf
from tools import gaussian_to_categorical

# BUILD THE INFERENCE NETWORK
def single_action_weights(Ns,No,
                        feedback_noise_std = 0.1,
                        N_up_actions= 1,N_useless_actions= 10,N_down_actions= 1,
                        naive_action_mapping = None,
                        naive_feedback_mapping = None,
                        naive_d_mapping = None,
                        decay_probability = 0.5,
                        trans_probability = 0.9,clamp_gaussian=True):
    """ 
    MDP weights for a mental training paradigm 
    with a single latent cognitive dimension. 
    """
    # In this "step" paradigm
    # We assume that each successive mental state can be achieved through
    # taking a specific action. To perform the task correctly, the subject
    # has to learn a specific state -> action decision rule !
    Nu = N_up_actions + N_down_actions + N_useless_actions

    b0 = np.zeros((Ns,Ns,Nu))

    # Simulate cognitive resting state
    for action in range(Nu):
        for from_state in range(Ns):
            if(from_state-1>=0):
                b0[from_state-1,from_state,action] = decay_probability
                b0[from_state,from_state,action] = 1.0 - decay_probability
            else : 
                b0[from_state,from_state,action] = 1.0

    for action in range(0,N_up_actions,1):
        # The first N_up_actions allow us to go from state i to i+1 along the only dim
        for from_state in range(Ns):
            b0[:,from_state,action] = np.zeros((Ns,))
            if (from_state<Ns-1):
                b0[from_state+1,from_state,action] = trans_probability  # probability of transition
                b0[from_state,from_state,action] = 1.0 - trans_probability 
            else :
                b0[from_state,from_state,action] = 1.0
    for action in range(N_up_actions,N_up_actions+N_down_actions,1):
        # The first N_up_actions allow us to go from state i to i+1 along the only dim
        for from_state in range(Ns):
            b0[:,from_state,action] = np.zeros((Ns,))
            if (from_state>1):
                b0[from_state-1,from_state,action] = trans_probability  # probability of transition
                b0[from_state,from_state,action] = 1.0 - trans_probability 
            else :
                b0[from_state,from_state,action] = 1.0
    b = [b0]

    if actynf.isField(naive_action_mapping):
        # naive action mapping is a list of 2 terms :
        # Term 0 : how much of the true mapping is known / before the trial over ALL transitions
        # Term 1 : initial confidence linked to the whole mapping
        b[0] = actynf.normalize(np.ones(b[0].shape) + naive_action_mapping[0]*b[0])*naive_action_mapping[1]
    #______________________________________________________________________________________

    a0 = np.zeros((No,Ns))
    for state in range(Ns):
        x_state = float(state)/(Ns-1) # State indicator between 0 and 1
        # print(x_state,x_state*(No-1))

        # print(np.round(gaussian_to_categorical(np.zeros((3,)),x_state*(No-1),0.25,option_clamp=False),2))
        a0[:,state] = gaussian_to_categorical(a0[:,state],x_state*(No-1),feedback_noise_std,option_clamp=clamp_gaussian)
    a = [a0]
    if actynf.isField(naive_feedback_mapping):
        # naive action mapping is a list of 2 terms :
        # Term 0 : how much of the true mapping is known / before the trial over ALL transitions
        # Term 1 : initial confidence linked to the whole mapping
        a[0] = actynf.normalize(np.ones(a[0].shape) + naive_feedback_mapping[0]*a[0])*naive_feedback_mapping[1]

    kC = 2.0
    c = [np.linspace(0,kC*(No-1),No)]

    d = [np.zeros((Ns,))]
    # Assume that the starting mental state is always relatively low
    starting_prop = 0.3
    d[0][:int(Ns*starting_prop+1)] += 1.0
    if actynf.isField(naive_d_mapping):
        # naive action mapping is a list of 2 terms :
        # Term 0 : how much of the true mapping is known / before the trial over ALL transitions
        # Term 1 : initial confidence linked to the whole mapping
        d[0] = actynf.normalize(np.ones(d[0].shape) + naive_d_mapping[0]*d[0])*naive_d_mapping[1]

    e = np.ones((Nu,))

    u = np.array(range(Nu))
    return a,b,c,d,e,u

def subject_model(T,Th,
                  Ns,No,feedback_std,
                  N_up_actions,N_useless_actions,N_down_actions,
                  action_map,feedback_map,d_map,
                  decay_probability=0.5,
                  action_effect_probability=0.9,
                  clamp_gaussian = True,
                  learning_space_structure = actynf.LINEAR,gen_temp = 3.0):
    
    a,b,c,d,e,u  = single_action_weights(Ns,No,
        feedback_noise_std=feedback_std,
        N_up_actions= N_up_actions,N_useless_actions= N_useless_actions,N_down_actions= N_down_actions,
        naive_action_mapping=action_map,
        naive_feedback_mapping=feedback_map,
        naive_d_mapping= d_map,
        decay_probability=decay_probability,
        trans_probability=action_effect_probability,
        clamp_gaussian=clamp_gaussian)
    
    model_layer = actynf.layer("subject_model","model",
                 a,b,c,d,e,u,
                 T,Th)
    
    model_layer.learn_options.learn_a = True
    model_layer.learn_options.learn_b = True
    model_layer.learn_options.learn_c = False
    model_layer.learn_options.learn_d = True
    model_layer.learn_options.learn_e = False

    model_layer.learn_options.assume_state_space_structure = learning_space_structure
    model_layer.learn_options.generalize_fadeout_function_temperature = gen_temp
        # Quick exponential decay for generalization
    return model_layer

def neurofeedback_process(T,Ns,No,feedback_std,
                  N_up_actions,N_useless_actions,N_down_actions,
                  decay_probability = 0.5,
                  action_effect_probability = 0.9,
                  clamp_gaussian = True):
    a,b,c,d,e,u  = single_action_weights(Ns,No,
        feedback_noise_std=feedback_std,
        N_up_actions= N_up_actions,N_useless_actions= N_useless_actions,N_down_actions= N_down_actions,
        decay_probability=decay_probability,
        trans_probability=action_effect_probability,
        clamp_gaussian = clamp_gaussian)
    process_layer = actynf.layer("process","process",
                 a,b,c,d,e,u,
                 T)
    return process_layer

def neurofeedback_training_one_action(T,Th,
                           Ns_subj,Ns_proc,
                           No_subj,No_proc,
                           feedback_std_subj,feedback_std_proc,
                           N_up_actions,N_useless_actions,N_down_actions,
                           action_beliefs,
                           perception_beliefs,
                           initial_state_beliefs,
                           decay_p,action_effect_p,
                           clamp_gaussian = True,
                           learning_space_structure=actynf.LINEAR,
                           gen_temp=3.0):
    process = neurofeedback_process(T,Ns_proc,No_proc,feedback_std_proc,
                                    N_up_actions,N_useless_actions,N_down_actions,
                                    decay_probability=decay_p,action_effect_probability=action_effect_p,
                                    clamp_gaussian=clamp_gaussian)
    
    model = subject_model(T,Th,Ns_subj,No_subj,feedback_std_subj,
                          N_up_actions,N_useless_actions,N_down_actions,
                          action_beliefs,
                          perception_beliefs,
                          initial_state_beliefs,
                          decay_probability = decay_p,action_effect_probability=action_effect_p,
                          clamp_gaussian=clamp_gaussian,learning_space_structure=learning_space_structure,
                          gen_temp=gen_temp)

    process.inputs.u = actynf.link(model,lambda x : x.u)
    model.inputs.o = actynf.link(process, lambda x : np.array([np.round(No_subj*(x.o[0]/No_proc))]).astype(int))

    net = actynf.layer_network([process,model],"neurofeedback_training_net")
    return net
