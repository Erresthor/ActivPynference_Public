import numpy as np, matplotlib.pyplot as plt, sys,os

import actynf
from actynf.layer.model_layer import mdp_layer
from actynf.layer.layer_link import establish_layerLink
from actynf.architecture.network import network

def feedback_based_on_inference_process(T = 10,seed=None):
    Thor = 2
    # A noisy feedback :
    a1 = np.array([[0.8,0.2,0.0,0.0,0.0],
                   [0.2,0.6,0.2,0.0,0.0],
                   [0.0,0.2,0.6,0.2,0.0],
                   [0.0,0.0,0.2,0.6,0.2],
                   [0.0,0.0,0.0,0.2,0.8]])
    # A very noisy feedback :
    a1 = np.array([[0.7,0.3,0.1,0.0,0.0],
                   [0.2,0.4,0.2,0.1,0.0],
                   [0.1,0.2,0.4,0.2,0.1],
                   [0.0,0.1,0.2,0.4,0.2],
                   [0.0,0.0,0.1,0.3,0.7]])
    # a1 = np.eye(5)
    No = a1.shape[0]
    a = [a1]

    Ns = 5
    b0 = np.zeros((Ns,Ns,1))  
    b = [b0]

    c = np.zeros((No,))
    
    d = np.array([1.0,0,0,0,0])

    e = np.ones((1,))
    u = np.array([0])
    
    process = mdp_layer("process","process",a,b,c,d,e,u,T,Thor,seed)
    process.hyperparams.process_definite_state_to_obs = False 
            # The observations are generated directly from the subject representation of 
            # the current state (x_d)
            # Not from definite states (x)
    process.hyperparams.process_definite_state_to_state = False
            # Has no effect on the computations but allow the value
            # that generated the observation to be saved directly
            
    return process

def neurofeedback_subject_model(T = 10,seed=None):
    action_selection_temperature = 32
    Thor = 2
    Ns = 5

    # A noisy feedback :
    a1 = np.array([[0.8,0.2,0.0,0.0,0.0],
                   [0.2,0.6,0.2,0.0,0.0],
                   [0.0,0.2,0.6,0.2,0.0],
                   [0.0,0.0,0.2,0.6,0.2],
                   [0.0,0.0,0.0,0.2,0.8]])
    # a1 = 0.1*(np.eye(5) + np.ones((Ns,Ns)))
    a = [a1]
    Nactions = 5

    # Subject perception of mental actions
    b0 = np.zeros((Ns,Ns,Nactions))  
    # Noisy stay the same
    for k in range(0,3):
        b0[:,:,k] = np.array( [[0.8,0.2,0.0,0.0,0.0],
                               [0.2,0.6,0.2,0.0,0.0],
                               [0.0,0.2,0.6,0.2,0.0],
                               [0.0,0.0,0.2,0.6,0.2],
                               [0.0,0.0,0.0,0.2,0.8]])
    # Noisy up
    b0[:,:,3] = np.array([[0.2,0.0,0.0,0.0,0.0],
                          [0.6,0.2,0.0,0.0,0.0],
                          [0.2,0.6,0.2,0.0,0.0],
                          [0.0,0.2,0.6,0.2,0.2],
                          [0.0,0.0,0.2,0.8,0.8]])

    # Noisy down
    b0[:,:,4] = np.array([[0.8,0.8,0.2,0.0,0.0],
                          [0.2,0.2,0.6,0.2,0.0],
                          [0.0,0.0,0.2,0.6,0.2],
                          [0.0,0.0,0.0,0.2,0.6],
                          [0.0,0.0,0.0,0.0,0.2]])
    b = [b0]
    print(b)

    b = [0.1*np.ones((Ns,Ns,Nactions))]

    c = [np.array([-2,0,2,4,6])]

    d = [np.array([0.2,0.2,0.2,0.2,0.2])]

    e = np.ones((Nactions,))

    u = np.array(range(Nactions))
    subject_model = mdp_layer("subject_model","model",a,b,c,d,e,u,T,Thor,seed)

    
    # Here, we give a few hyperparameters guiding the beahviour of our agent :
    subject_model.hyperparams.alpha = action_selection_temperature # action precision : 
        # for high values the mouse will always perform the action it perceives as optimal, with very little exploration 
        # towards actions with similar but slightly lower interest



    subject_model.learn_options.eta = 1 # learning rate (shared by all channels : a,b,c,d,e)
    subject_model.learn_options.learn_a = False  # The agent learns the reliability of the clue
    subject_model.learn_options.learn_b = True # The agent does not learn transitions
    subject_model.learn_options.learn_d = True  # The agent has to learn the initial position of the cheese
    subject_model.learn_options.backwards_pass = True  # When learning, the agent will perform a backward pass, using its perception of 
                                               # states in later trials (e.g. I saw that the cheese was on the right at t=3)
                                               # as well as what actions it performed (e.g. and I know that the cheese position has
                                               # not changed between timesteps) to learn more reliable weights (therefore if my clue was
                                               # a right arrow at time = 2, I should memorize that cheese on the right may correlate with
                                               # right arrow in general)
    subject_model.learn_options.memory_loss = 0
                                            # How many trials will be needed to "erase" 50% of the information gathered during one trial
                                            # Used during the learning phase
    # subject_model.hyperparams.cap_state_explo = 3
    # subject_model.hyperparams.cap_action_explo = 2

    return subject_model

if __name__ == '__main__':  
    print(actynf.__version__)
    T = 25
    model = neurofeedback_subject_model(T)
    process = feedback_based_on_inference_process(T)

    link1 = establish_layerLink(process,model,["o","o"])
    link2 = establish_layerLink(model,process,["s_d","s_d"])
    print(link1)
    print(link2)
    broken_markov_blanket_net = network([model,process],"broken_markov_blanket_network")


    Ntrials = 10
    for sample in range(Ntrials):
        
        broken_markov_blanket_net.run(verbose=False)

        print("process")
        print(np.round(process.STM.x_d,2))
        print(process.STM.o)
        print("model")
        print(np.round(model.STM.x_d,2))
        print(model.STM.u)

        fig,axes = plt.subplots(2,sharex=True)

        fig.suptitle("Trial " + str(sample), fontsize=16)
        infer_model = model.STM.x_d
        # axes[0].set_ylim([4.5,-0.5])
        axes[0].imshow(infer_model,aspect='auto',interpolation='nearest',vmin=0,vmax=1)#,extent=[0,T+1,0,1]
        axes[1].scatter(np.linspace(0,T-1,T),process.STM.o[0,:],color='black',marker=".",s=150)
        axes[1].imshow(process.STM.o_d,aspect='auto',interpolation='nearest',vmin=0,vmax=1)#,extent=[0,T+1,0,1]
        

        # LABELS
        axes[0].set_ylabel("Subject perceiverd state")
        axes[1].set_ylabel("Provided feedback \n (based on subject perception)")
        axes[1].set_xlabel("Trials")
        
        for ax in axes :
            ax.invert_yaxis()
            labels = [item.get_text() for item in ax.get_yticklabels()]
            for i in range(len(labels)):
                labels[i] = ""
            labels[1] = 'LOW'
            labels[-2] = 'HIGH'
            ax.yaxis.set_ticklabels(labels)

    plt.show()




