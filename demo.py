A_obs_mental = np.zeros((Ns[0],Ns[0],Ns[1]))
    # When attentive, the feedback is modelled as perfect :
A_obs_mental[:,:,0] = np.array([[1,0,0,0,0],
                                [0,1,0,0,0],
                                [0,0,1,0,0],
                                [0,0,0,1,0],
                                [0,0,0,0,1]])