import numpy as np
import matplotlib.pyplot as plt

from actynf.layer.model_layer import mdp_layer
from actynf.layer.layer_link import establish_layerLink
from actynf.architecture.network import network
from actynf.base.function_toolbox import normalize

def get_T_maze_gen_process(pinit,pHA,pWin):
    """
    pinit : prob of reward initial position being left / right
    pHA : probability of clue giving the correct index position
    pWin : probability of winning if we are in the correct position
    """
    print("T-maze gen. process set-up ...  ",end='')
    T = 3

    d = [np.array([pinit,1-pinit]),np.array([1,0,0,0])]
    Ns = [arr.shape[0] for arr in d]
    
    # Transition matrixes between hidden states ( = control states)
    #a. Transition between context states --> The agent cannot act so there is only one :
    B_context_states = np.array([[[1],[0]],
                                 [[0],[1]]])
    #b. Transition between behavioural states --> 4 actions
    B_behav_states = np.zeros((Ns[1],Ns[1],Ns[1]))
    B_behav_states[0,:,0] = 1          # - 0 --> Move to start from any state
    B_behav_states[1,:,1] = 1          # - 1 --> Move to hint from any state
    B_behav_states[2,:,2] = 1          # - 2 --> Move to choose left from any state
    B_behav_states[3,:,3] = 1          # - 3 --> Move to choose right from any state
    b = [B_context_states, B_behav_states]


    #Mapping from states to observed hints, accross behaviour states (non represented)
    #
    # [ .  . ]  No hint
    # [ .  . ]  Machine Left Hint            Rows = observations
    # [ .  . ]  Machine Right Hint
    # Left Right
    # Columns = context state
    A_obs_hints = np.zeros((3,Ns[0],Ns[1]))
    A_obs_hints[0,:,:] = 1
    A_obs_hints[:,:,1] = np.array([[0,0],
                             [pHA, 1-pHA],
                             [1-pHA,pHA]]) # Behaviour ste "hint" gives an observed hint
    
    #Mapping from states to outcome (win / loss / null), accross behaviour states (non represented)
    #
    # [ .  . ]  Null
    # [ .  . ]  Win           Rows = observations
    # [ .  . ]  Loss
    # Left Right
    # Columns = context state
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
    
    #Mapping from behaviour states to observed behaviour
    #
    # [ .  .  .  .] start
    # [ .  .  .  .] hint
    # [ .  .  .  .] choose left         Row = Behaviour state
    # [ .  .  .  .] choose right
    #  s   h  l  r
    #
    # 3rd dimension = observed behaviour
    # The 2nd dimension maps the dependance on context state
    A_obs_behaviour = np.zeros((Ns[1],Ns[0],Ns[1]))
    for i in range (Ns[1]) :
        A_obs_behaviour[i,:,i] = np.array([1,1])
    a = [A_obs_hints,A_obs_outcome,A_obs_behaviour]

    No = [ai.shape[0] for ai in a]

    c = [np.zeros((No[0],T)),np.zeros((No[1],T)),np.zeros((No[2],T))]
    
    u = np.array([[0,0],[0,1],[0,2],[0,3]]).astype(np.int)
    
    #Habits
    e = np.ones((u.shape[0],))

    layer = mdp_layer("T-maze_environment","process",a,b,c,d,e,u,T)
    print("Done.")
    return layer


def get_new_T_maze_gen_process(pinit,pHA,pWin):
    """
    pinit : prob of reward initial position being left / right
    pHA : probability of clue giving the correct index position
    pWin : probability of winning if we are in the correct position
    """
    print("T-maze gen. process set-up ...  ",end='')
    T = 3

    d = [np.array([pinit,1-pinit]),np.array([1,0,0,0])]
    Ns = [arr.shape[0] for arr in d]
    
    # Transition matrixes between hidden states ( = control states)
    #a. Transition between context states --> The agent cannot act so there is only one :
    B_context_states = np.array([[[1],[0]],
                                 [[0],[1]]])
    #b. Transition between behavioural states --> 4 actions
    B_behav_states = np.zeros((Ns[1],Ns[1],Ns[1]))
    # - 0 --> Move to start from any state
    B_behav_states[0,:,0] = 1          

    # - 1 --> Move to hint from start, else go to start
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


    #Mapping from states to observed hints, accross behaviour states (non represented)
    #
    # [ .  . ]  No hint
    # [ .  . ]  Machine Left Hint            Rows = observations
    # [ .  . ]  Machine Right Hint
    # Left Right
    # Columns = context state
    A_obs_hints = np.zeros((3,Ns[0],Ns[1]))
    A_obs_hints[0,:,:] = 1
    A_obs_hints[:,:,1] = np.array([[0,0],
                             [pHA, 1-pHA],
                             [1-pHA,pHA]]) # Behaviour ste "hint" gives an observed hint
    
    #Mapping from states to outcome (win / loss / null), accross behaviour states (non represented)
    #
    # [ .  . ]  Null
    # [ .  . ]  Win           Rows = observations
    # [ .  . ]  Loss
    # Left Right
    # Columns = context state
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
    
    #Mapping from behaviour states to observed behaviour
    #
    # [ .  .  .  .] start
    # [ .  .  .  .] hint
    # [ .  .  .  .] choose left         Row = Behaviour state
    # [ .  .  .  .] choose right
    #  s   h  l  r
    #
    # 3rd dimension = observed behaviour
    # The 2nd dimension maps the dependance on context state
    A_obs_behaviour = np.zeros((Ns[1],Ns[0],Ns[1]))
    for i in range (Ns[1]) :
        A_obs_behaviour[i,:,i] = np.array([1,1])
    a = [A_obs_hints,A_obs_outcome,A_obs_behaviour]

    No = [ai.shape[0] for ai in a]

    c = [np.zeros((No[0],T)),np.zeros((No[1],T)),np.zeros((No[2],T))]
    
    u = np.array([[0,0],[0,1],[0,2],[0,3]]).astype(np.int)
    
    #Habits
    e = np.ones((u.shape[0],))

    layer = mdp_layer("T-maze_environment","process",a,b,c,d,e,u,T)
    print("Done.")
    return layer


def get_T_maze_model(true_process_layer,la,rs,T_horizon,K = 0.1):
    """
    pinit : prob of reward initial position being left / right
    pHA : probability of clue giving the correct index position
    pWin : probability of winning if we are in the correct position
    """
    print("T-maze gen. model set-up ...  ",end='')
    T = 3

    d = [np.array([0.25,0.25]),np.array([1,0,0,0])]

    Ns = [arr.shape[0] for arr in d]
    
    # Transition matrixes between hidden states ( = control states)
    b=[]
    for b_fac_proc in (true_process_layer.b):
        b.append(np.copy(b_fac_proc)*200)


    a = []
    for a_mod_proc in (true_process_layer.a):
        a.append(np.copy(a_mod_proc)*200)
    a[0][:,:,1] = K*np.array([[0,0],
                            [0.25,0.25],
                            [0.25,0.25]])  
            # Have to learn the reliability of the indice
    # a[0][:,:,1] = np.array([[0,0],
    #                         [0.25+1,0.25],
    #                         [0.25,0.25+1]])  



    No = [ai.shape[0] for ai in a]

    C_hints = np.zeros((No[0],T))
    C_win_loss = np.zeros((No[1],T))
    C_win_loss = np.array([[0,0,0],     #null
                           [0,rs,rs/2.0],  #win
                           [0,-la,-la]]) #loss
    C_observed_behaviour = np.zeros((No[2],T))
    c = [C_hints,C_win_loss,C_observed_behaviour]
    
    u = np.array([[0,0],[0,1],[0,2],[0,3]]).astype(np.int)
    
    #Habits
    e = np.ones((u.shape[0],))

    layer = mdp_layer("mouse_model","model",a,b,c,d,e,u,T,T_horiz=T_horizon)

    layer.hyperparams.alpha = 32 # action precision
    layer.hyperparams.beta = 1    # policy precision
    layer.hyperparams.erp = 4  # degree of belief resetter at each time point
             # ( 1 means no reset bet. time points, higher values mean more loss in confidence)
    layer.hyperparams.chi = 1/64  # Occam window updates

    layer.learn_options.eta = 1
    layer.learn_options.learn_b = False
    print("Done.")
    return layer

def get_t_maze_network(pinit,pHA,pWin,la,rs,T_horizon,K,new =True):
    if (new):
        maze_process = get_new_T_maze_gen_process(pinit,pHA,pWin)
    else :
        maze_process = get_T_maze_gen_process(pinit,pHA,pWin)
    print(maze_process)
    maze_model = get_T_maze_model(maze_process,la,rs,T_horizon,K)
    print(maze_model)
    link_act = establish_layerLink(maze_model,maze_process,["u","u"])
    link_obs = establish_layerLink(maze_process,maze_model,["o","o"])
    tmaze_net = network([maze_process,maze_model],"t-maze")
    return tmaze_net

def example_tmaze_plot():
    pinit = 0.5 # Initial side
    pinit2 = 0.95

    pHA = 0.95   # P of clue showing the right side
    pWin = 0.98 # P of reward if you get the right side
    la = 1
    rs = 2
    Thorizon = 2
    K = 1 # 0.5

    tmaze_net = get_t_maze_network(pinit,pHA,pWin,la,rs,Thorizon,K)
    print(tmaze_net)

    N = 60
    Nswitch = 10

    img_process = np.zeros((2,N+1))
    img_model = np.zeros((2,N+1))
    infer_model = np.zeros((N*3,))
    agent_actions = np.zeros((2,N))
    img_act = np.zeros((3,N))

    img_process[:,0] = tmaze_net.layers[0].d[0]
    img_model[:,0] = normalize(tmaze_net.layers[1].d[0])
    

    fig,axes = plt.subplots(3,sharex=True)
    
    for i in range(N):
        if i == Nswitch:
            tmaze_net.layers[0].d[0] = np.array([pinit2,1-pinit2])
        tmaze_net.run(verbose=False)
        print(str(tmaze_net.layers[1].STM.u_d))


        img_process[:,i+1] = tmaze_net.layers[0].d[0]
        img_model[:,i+1] = normalize(tmaze_net.layers[1].d[0])
        img_act[:,i] = tmaze_net.layers[1].STM.u_d[1:,0]

        reward_dim = np.sum(tmaze_net.layers[1].STM.x_d,axis = 1)
        infer_model[i*3:(i*3+3)] = reward_dim[0,:]
        agent_actions[:,i] = tmaze_net.layers[1].STM.u
    img_act[[0, 1],:] = img_act[[1, 0],:]



    marker='.'
    markersize=100
    axes[0].imshow(img_process,aspect='auto',interpolation='nearest',extent=[0,N+1,0,1],vmin=0,vmax=1)
    axes[1].scatter(np.linspace(1,N+1,3*N),(infer_model),color='black',marker=".",s=markersize)
    axes[2].axvline(0,color='black')
    for trial in range(agent_actions.shape[1]) :
        axes[2].axvline(trial+1,color='black')
        acts = agent_actions[:,trial]
        pos = 0
        for act in acts :
            if (act==2):
                axes[2].scatter([trial+1+pos],[1],color='red',marker='^',s=markersize)
                break
            elif (act==3):
                axes[2].scatter([trial+1+pos],[0],color='red',marker='v',s=markersize)
                break
            elif (act==1):
                axes[2].scatter([trial+1+pos],[0.5],color='green',marker='>',s=markersize)
            elif (act==0):
                axes[2].scatter([trial+1+pos],[0.5],color='orange',marker='<',s=markersize)
            pos = pos + 1.0/3.0
    
    axes[2].imshow(img_act,aspect='auto',interpolation='nearest',extent=[0,N+1,0,1],vmin=0,vmax=1)
    axes[1].imshow(img_model,aspect='auto',interpolation='nearest',extent=[0,N+1,0,1],vmin=0,vmax=1)


    for ax in axes:
        ax.set_ylim([-0.1,1.1])
        ax.set_xlim([0,N+1])
    # axes[1].set_ylim([-0.1,1.1])
    # axes[2].set_ylim([-0.1,1.1])
    axes[0].set_ylabel("Context")
    axes[1].set_ylabel("State model")
    axes[2].set_ylabel("Action selection")
    axes[2].set_xlabel("Trials")
    plt.show()