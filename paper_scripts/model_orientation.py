
def transition_weights_centered(Ns,
                   N_up_actions,N_down_actions,N_neutral_actions,
                   decay_probability,transition_probability,
                   resting_state_per_factor):
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

        
        resting_state = resting_state_per_factor[f] # What is the resting state for the subject ?
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
