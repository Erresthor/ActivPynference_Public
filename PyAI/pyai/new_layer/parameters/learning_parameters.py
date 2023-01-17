class learning_parameters :
    def __init__(self):
        self.T_horizon = 1
        self.update_frequency = 1 # ]0,1] --> At most, you can be updated once every loop, at worst, only once in total
                                    # To be implemented
        
        self.learn_during_trial = False
        self.memory_loss = 0

        self.learn_a = True
        self.learn_b = True
        self.learn_c = False
        self.learn_d = True
        self.learn_e = False