import numpy as np
import actynf


def build_training_process(Ns,p_up,p_low,kas):
    """
    pinit : prob of reward initial position being left / right
    pHA : probability of clue giving the correct index position
    pWin : probability of winning if we are in the correct position

    This function returns a mdp_layer representing the t-maze environment.
    """
    print("T-maze gen. process set-up ...  ",end='')

    T = 10  # The trials are made of 3 timesteps (starting step + 2 others)

    Ns = 5 
    Np = 6
    
    # Initial situation
    d = [np.zeros((Ns,))]
    d[0][0] = 1.0
    
    
    b = np.zeros((Ns,Ns,Np))
    for u in range(Np):
        for s in range(Ns):
            if s >0:
                b[s-1,s,u] = p_low
                b[s,s,u] = 1-p_low
            else : 
                b[s,s,u] = 1.0
        try :
            b[u+1,u,u] = p_up
            b[u,u,u] = 1.0 - p_up
            b[u-1,u,u] = 0.0
        except:
            lol = "lol"
    b = actynf.normalize([b])

    # Active Inference also revolves around a state-observation correspondance that we describe here :
    [ka1,ka2] = kas
    a1 = (ka1)*np.eye(Ns)+(1-ka1)*np.ones((Ns,Ns))
    a2 = (ka2)*np.eye(Ns)+(1-ka2)*np.ones((Ns,Ns))
    a = actynf.normalize([a1,a2])
    
    # ... as well as the allowable transitions the mouse can choose :
    u = np.linspace(0,Np-1,Np).astype(int)
    
    # Habits
    e = np.ones((u.shape[0],))

    # The environment has been well defined and we may now build a mdp_layer using the following constructor : 
    # layer = mdp_layer("T-maze_environment","process",a,b,c,d,e,u,T)
    #     mdp_layer(name of the layer,process or model, a,b,c,d,e,u,T)
    print("Done.")
    return a,b,d,u


def build_subject_model(Ns,
                        a_str,kas_subj,
                        process_b,kb_subj,b_str,
                        process_u,
                        kd,
                        rs):
    """
    true_process_layer : the mdp_layer object where the tmaze environment has been defined
    la : how much the mouse is afraid of adverse outcomes (>0)
    rs : how much the mouse wants to observe cheese (>0)
    T_horizon : how much into the future the mouse will plan before picking its next action
    initial_clue_confidence : how much the mouse knows about the clue reliability
    """
    print("T-maze gen. model set-up ...  ",end='')
    T = 3

    #  The mouse knows where it stands in the maze initially, but it doesn't know where the cheese will spawn : this is something that
    # it will need to learn !
    one_hot = np.zeros(Ns,)
    one_hot[0] = 1.0
    d = [kd*one_hot + (1.0-kd)*np.ones((Ns,))]
    

    
    # Transition matrixes between hidden states ( = control states)
    b=[]
    for b_fac_proc in (process_b):
        b.append(b_str*(kb_subj*b_fac_proc + (1.0-kb_subj)*np.ones_like(b_fac_proc)))
    # b[1]= np.ones
    # The mouse knows how its action will affect the general situation. The mouse does not need
    # to learn that element . Be aware that too much uncertainty in some situations may prove hard to resolve for our
    # artifical subjects.


    [ka1,ka2] = kas_subj
    [astr1,astr2] = a_str
    a1 = astr1*((ka1)*np.eye(Ns)+(1-ka1)*np.ones((Ns,Ns)))
    a2 = astr2*((ka2)*np.eye(Ns)+(1-ka2)*np.ones((Ns,Ns)))
    a = [a1,a2]


    # Finally, the preferences of the mouse are governed by the experimenter through the rs/la weights.
    C0 = np.linspace(0,rs,Ns)
    C1 = np.zeros((Ns,))
    c = [C0,C1]
    
    # The allowable actions have been defined earlier
    u = process_u
    
    # Habits
    e = np.ones((u.shape[0],))
    print("Done !")
    return a,b,c,d,e,u