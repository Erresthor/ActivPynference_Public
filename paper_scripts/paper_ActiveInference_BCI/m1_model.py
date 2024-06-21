import numpy as np

import actynf
from tools import gaussian_to_categorical

def observation_matrix(Ns,No,feedback_noise_std,k1a,uniform_eps= 0.01):
    a0 = np.zeros((No,Ns))
    for state in range(Ns):
        x_state = float(state)/(Ns-1) # State indicator between 0 and 1
        a0[:,state] = gaussian_to_categorical(a0[:,state],x_state*(No-1),feedback_noise_std,option_clamp=False)
    
    a = k1a*actynf.normalize(a0 + uniform_eps*np.ones(a0.shape))
    return a

# BUILD THE INFERENCE NETWORK
def m1_weights(Ns,No,
            feedback_noise_std = 0.1,
            factor_useless_actions= 2,
            k1b = None, epsilon_b = 0.0,
            k1a = None, epsilon_a = 0.0,
            k1d = None, epsilon_d = 1.0,
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

    if actynf.isField(k1b):
        adequate_prior_knowledge = 0.0  # Fixed for now
        b0 = k1b*actynf.normalize(adequate_prior_knowledge*b0 + epsilon_b*np.ones(b0.shape))
    b = [b0]

    # Observation matrix from the above function
    if (actynf.isField(k1a)):
        a0 = observation_matrix(Ns,No,feedback_noise_std,k1a,epsilon_a)
    else :
        a0 = observation_matrix(Ns,No,feedback_noise_std,1.0,0.0)
    a = [a0]

    kC = 2.0
    c = [np.linspace(0,kC*(No-1),No)]

    d0 = np.zeros((Ns,))
    # Let the subject assume that the starting mental state is always relatively low
    starting_prop = 0.3
    d0[:int(Ns*starting_prop+1)] += 1.0

    if actynf.isField(k1d):
        # naive action mapping is a list of 2 terms :
        # Term 0 : how much of the true mapping is known / before the trial over ALL transitions
        # Term 1 : initial confidence linked to the whole mapping
        d0 = k1d*actynf.normalize(d0 + epsilon_d * np.ones(d0.shape))
    d = [d0]
    
    e = np.ones((Nu,))

    u = np.array(range(Nu))

    return a,b,c,d,e,u

def subject_model(T,Th,
                  Ns,No,feedback_std,
                  k_useless_actions,
                  k1b, epsilon_b,
                  k1a, epsilon_a,
                  k1d, epsilon_d,
                  decay_probability=0.5,
                  action_effect_probability=0.9,
                  clamp_gaussian = True,asit=1.0,learn_a=True):
    
    a,b,c,d,e,u  = m1_weights(Ns,No,
        feedback_noise_std=feedback_std,
        factor_useless_actions=k_useless_actions,
        k1b=k1b,epsilon_b=epsilon_b,
        k1a=k1a,epsilon_a=epsilon_a,
        k1d=k1d,epsilon_d=epsilon_d,
        decay_probability=decay_probability,
        up_probability=action_effect_probability,
        clamp_gaussian=clamp_gaussian)
    
    model_layer = actynf.layer("subject_model","model",
                 a,b,c,d,e,u,
                 T,Th)
    
    model_layer.learn_options.learn_a = learn_a

    model_layer.learn_options.learn_b = True
    model_layer.learn_options.backwards_pass = False 
        # Weither or not to perform a a posteriori 
        # belief update using latter states inferences
        # Seems bugged for now, be careful

    model_layer.learn_options.learn_c = False
    model_layer.learn_options.learn_d = True
    model_layer.learn_options.learn_e = False

    model_layer.hyperparams.alpha = asit 
        # How much noise in the eventual action selection

    return model_layer

def neurofeedback_process(T,Ns,No,feedback_std,
                  k_useless_actions,
                  decay_probability = 0.5,
                  action_effect_probability = 0.9,
                  clamp_gaussian = True):
    
    a,b,c,d,e,u  = m1_weights(Ns,No,
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
                           k1b,epsilon_b,
                           k1a,epsilon_a,
                           k1d,epsilon_d,
                           decay_p,action_effect_p,
                           clamp_gaussian = True,asit=32,learn_a=True):
    
    process = neurofeedback_process(T,Ns_proc,No_proc,feedback_std_proc,k_useless_actions,
                                    # NO PRIOR WEIGHTS HERE
                                    decay_probability=decay_p,action_effect_probability=action_effect_p,
                                    clamp_gaussian=clamp_gaussian)
    
    model = subject_model(T,Th,Ns_subj,No_subj,feedback_std_subj,k_useless_actions,
                          k1b,epsilon_b,
                          k1a,epsilon_a,
                          k1d,epsilon_d,
                          decay_probability = decay_p,action_effect_probability=action_effect_p,
                          clamp_gaussian=clamp_gaussian,asit=asit,
                          learn_a=learn_a)

    process.inputs.u = actynf.link(model,lambda x : x.u)
    model.inputs.o = actynf.link(process, lambda x : np.array([np.round(No_subj*(x.o[0]/No_proc))]).astype(int))

    net = actynf.layer_network([process,model],"neurofeedback_training_net")
    return net
