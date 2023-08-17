class learning_parameters :
    def __init__(self):
        self.learn_during_trial = False
        
        self.eta = 1     
            # learning rate
        self.memory_loss = 0
            # memory loss rate

        self.backwards_pass = False
            # Should the agent perform a 
            # backward pass using its history of 
            # actions as well as its history of
            # state perception before learning :
            # If not, the agent only uses its direct perception
            # and may not use information it gathered later during the trial 
        

        self.learn_a = False
        self.learn_b = False
        self.learn_c = False
        self.learn_d = False
        self.learn_e = False