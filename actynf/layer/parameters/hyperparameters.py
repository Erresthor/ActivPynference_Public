class hyperparameters :
    def __init__(self):
        self.alpha = 32  # action precision
        self.beta = 1    # policy precision
        
        self.tau = 4     # update time constant (gradient descent)
        self.erp = 4     # update reset
        self.chi = 1/64  # Occam window updates
        self.zeta = 3    # Occam window policies
        

        # "Raw" parameters to prevent the tree search from being too computationally expensive
        self.cap_state_explo = None
        self.cap_action_explo = None
        # TODO : dynamic time-based tree search limiter (go as far as you can into
        # the future as long as it's below a certain time threshold)
        # self.global_time_cap_explo = 500 # ms

        self.b_novelty = False



        # /!\ messing up with the following settings may provide contradictory
        # informations to your network if you're not careful !
        self.process_definite_state_to_state = True
            # Weither or not the next hidden state distribution should be
            # computed from the last hidden state (rather than from the 
            # whole last state distribution)
            # (This option is used only when the layer is in process mode).
        self.process_definite_state_to_obs = True
            # Weither or not the current observation distribution should be
            # computed from the current hidden state (rather than from the 
            # whole current state distribution)
            # (This option is used only when the layer is in process mode).

        