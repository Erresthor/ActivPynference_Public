import numpy as np

import actynf
from tools import gaussian_to_categorical


# BUILD THE INFERENCE NETWORK
def climb_steps_weights(Ns,No,
                        feedback_noise_std = 0.1,
                        factor_useless_actions= 2,
                        naive_action_mapping = None,
                        naive_feedback_mapping = None,
                        naive_d_mapping = None,
                        decay_probability = 0.5,
                        up_probability = 0.9,clamp_gaussian=True):
    """ 
    MDP weights for a mental training paradigm 
    with a single latent cognitive dimension. 
    """
    N_useless_actions = int(factor_useless_actions*Ns)
    
    # In this "step" paradigm
    # We assume that each successive mental state can be achieved through
    # taking a specific action. To perform the task correctly, the subject
    # has to learn a specific state -> action decision rule !
    Nu = Ns + N_useless_actions
    b0 = np.zeros((Ns,Ns,Nu))
    for action in range(Nu):
        # here, action allows the subject to go from state (action-1) to state (action)
        # If taken from any other state, it does nothing !

        # IF THERE IS A COGNITIVE DECAY
        for from_state in range(Ns):
            if(from_state-1>=0):
                b0[from_state-1,from_state,action] = decay_probability
                b0[from_state,from_state,action] = 1.0 - decay_probability
            else : 
                b0[from_state,from_state,action] = 1.0

        # # IF THERE IS NO DECAY !
        # b0[:,:,action] = np.eye(Ns)

        if ((action-1)>=0) and (action < Ns): 
                # Action 0 has no purpose, as there is no -1th step
                # Actions >Ns have no effect on the cognitive state and account
                # for mental actions on other dims
            b0[:,action-1,action] = np.zeros((Ns,))
            b0[action,action-1,action] = up_probability  # probability of transition
            b0[action-1,action-1,action] = 1.0 - up_probability 
    b = [b0]
    if actynf.isField(naive_action_mapping):
        # naive action mapping is a list of 2 terms :
        # Term 0 : how much of the true mapping is known / before the trial over ALL transitions
        # Term 1 : initial confidence linked to the whole mapping
        b[0] = actynf.normalize(np.ones(b[0].shape) + naive_action_mapping[0]*b[0])*naive_action_mapping[1]

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
    # print(b[0][:,:,8])

def subject_model(T,Th,
                  Ns,No,feedback_std,
                  k_useless_actions,
                  action_map,feedback_map,d_map,
                  decay_probability=0.5,
                  action_effect_probability=0.9,
                  clamp_gaussian = True):
    
    a,b,c,d,e,u  = climb_steps_weights(Ns,No,
        feedback_noise_std=feedback_std,
        factor_useless_actions=k_useless_actions,
        naive_action_mapping=action_map,
        naive_feedback_mapping=feedback_map,
        naive_d_mapping= d_map,
        decay_probability=decay_probability,
        up_probability=action_effect_probability,
        clamp_gaussian=clamp_gaussian)
    model_layer = actynf.layer("subject_model","model",
                 a,b,c,d,e,u,
                 T,Th)
    
    model_layer.learn_options.learn_a = True
    model_layer.learn_options.learn_b = True
    model_layer.learn_options.learn_c = False
    model_layer.learn_options.learn_d = True
    model_layer.learn_options.learn_e = False

    return model_layer

def neurofeedback_process(T,Ns,No,feedback_std,
                  k_useless_actions,
                  decay_probability = 0.5,
                  action_effect_probability = 0.9,
                  clamp_gaussian = True):
    a,b,c,d,e,u  = climb_steps_weights(Ns,No,
        feedback_noise_std=feedback_std,
        factor_useless_actions=k_useless_actions,
        decay_probability=decay_probability,
        up_probability=action_effect_probability,
        clamp_gaussian = clamp_gaussian)
    process_layer = actynf.layer("process","process",
                 a,b,c,d,e,u,
                 T)
    return process_layer

def neurofeedback_training(T,Th,
                           Ns_subj,Ns_proc,
                           No_subj,No_proc,
                           feedback_std_subj,feedback_std_proc,
                           k_useless_actions,
                           action_beliefs,
                           perception_beliefs,
                           initial_state_beliefs,
                           decay_p,action_effect_p,
                           clamp_gaussian = True):
    process = neurofeedback_process(T,Ns_proc,No_proc,feedback_std_proc,k_useless_actions,
                                    decay_probability=decay_p,action_effect_probability=action_effect_p,
                                    clamp_gaussian=clamp_gaussian)
    model = subject_model(T,Th,Ns_subj,No_subj,feedback_std_subj,k_useless_actions,
                          action_beliefs,
                          perception_beliefs,
                          initial_state_beliefs,
                          decay_probability = decay_p,action_effect_probability=action_effect_p,
                          clamp_gaussian=clamp_gaussian)

    process.inputs.u = actynf.link(model,lambda x : x.u)
    model.inputs.o = actynf.link(process, lambda x : np.array([np.round(No_subj*(x.o[0]/No_proc))]).astype(int))

    net = actynf.layer_network([process,model],"neurofeedback_training_net")
    return net