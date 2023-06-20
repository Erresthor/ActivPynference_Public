class learning_parameters :
    def __init__(self):
        self.learn_during_trial = False
        self.memory_loss = 0

        self.eta = 1     # learning rate

        self.learn_a = True
        self.learn_b = True
        self.learn_c = False
        self.learn_d = True
        self.learn_e = False