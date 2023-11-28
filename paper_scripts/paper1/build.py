import numpy as np
import statistics as stat
import scipy.stats as scistats
import math

import actynf

from tools import gaussian_to_categorical

disc_gauss = gaussian_to_categorical(np.zeros(10,),2.3,1)
print(disc_gauss)
print(disc_gauss.shape[0])

def climb_steps_weights(Ns,No,
                        feedback_noise_std = 0.1,
                        factor_useless_actions= 2):
    """ 
    MDP weights for a mental training paradigm 
    with a single latent cognitive dimension. 
    """
    N_useless_actions = int(factor_useless_actions*Ns)
    decay_probability = 0.5
    up_probability = 0.9
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


    a0 = np.zeros((No,Ns))
    for state in range(Ns):
        x_state = float(state)/(Ns-1) # State indicator between 0 and 1
        # print(x_state,x_state*(No-1))

        # print(np.round(gaussian_to_categorical(np.zeros((3,)),x_state*(No-1),0.25,option_clamp=False),2))
        a0[:,state] = gaussian_to_categorical(a0[:,state],x_state*(No-1),feedback_noise_std,option_clamp=True)
    a = [a0]

    kC = 2.0
    c = [np.linspace(0,kC*(No-1),No)]

    d = [np.zeros((Ns,))]
    # Assume that the starting mental state is always relatively low
    starting_prop = 0.3
    d[0][:int(Ns*starting_prop+1)] += 1.0
    
    e = np.ones((Nu,))

    u = np.array(range(Nu))

    return a,b,c,d,e,u
    # print(b[0][:,:,8])

def subject_model(T,Th,
                  Ns,No,feedback_std,
                  k_useless_actions):
    a,b,c,d,e,u  = climb_steps_weights(Ns,No,
        feedback_noise_std=feedback_std,
        factor_useless_actions=k_useless_actions)
    model_layer = actynf.layer("subject_model","model",
                 a,b,c,d,e,u,
                 T,Th)
    
    model_layer.learn_options.learn_a = False
    model_layer.learn_options.learn_b = True
    model_layer.learn_options.learn_c = False
    model_layer.learn_options.learn_d = True
    model_layer.learn_options.learn_e = False

    return model_layer

def neurofeedback_process(T,Ns,No,feedback_std,
                  k_useless_actions):
    a,b,c,d,e,u  = climb_steps_weights(Ns,No,
        feedback_noise_std=feedback_std,
        factor_useless_actions=k_useless_actions)
    process_layer = actynf.layer("process","process",
                 a,b,c,d,e,u,
                 T)
    return process_layer

def neurofeedback_training(T,Th,
                           Ns_subj,Ns_proc,
                           No_subj,No_proc,
                           feedback_std_subj,feedback_std_proc,
                           k_useless_actions):
    process = neurofeedback_process(T,Ns_proc,No_proc,feedback_std_proc,k_useless_actions)
    model = subject_model(T,Th,Ns_subj,No_subj,feedback_std_subj,k_useless_actions)

    process.inputs.u = actynf.link(model,lambda x : x.u)
    model.inputs.o = actynf.link(process, lambda x : np.array([np.round(No_subj*(x.o[0]/No_proc))]))

    net = actynf.layer_network([process,model],"neurofeedback_training_net")
    return net

if __name__ == "__main__":
    my_net = neurofeedback_training(10,2,
                    5,5,
                    5,5,
                    0.3,0.1,
                    1.0)
