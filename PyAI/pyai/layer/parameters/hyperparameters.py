class hyperparameters :
    def __init__(self):
        self.alpha = 32 # action precision
        self.beta = 1    # policy precision
        
        self.tau = 4     # update time constant (gradient descent)
        self.erp = 4     # update reset
        self.chi = 1/64  # Occam window updates
        self.zeta = 3    # Occam window policies