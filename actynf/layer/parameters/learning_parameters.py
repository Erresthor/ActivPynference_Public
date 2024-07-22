from ...enums import NO_MEMORY_DECAY,NO_STRUCTURE
import numpy as np

class learning_parameters :
    def __init__(self,memory_decay = NO_MEMORY_DECAY,memory_loss = 0.0,
                 learn_backward_pass = True,use_backward_pass_to_learn_d=True,
                 state_structure_assumption = NO_STRUCTURE):
        
        self.learn_during_trial = False
        
        self.eta = 1     
            # learning rate, for now global

        self.decay_type = memory_decay
        self.memory_loss = memory_loss
            # memory loss rate

        self.backwards_pass = learn_backward_pass
            # Should the agent perform a 
            # backward pass using its history of 
            # actions as well as its history of
            # state perception before learning :
            # If not, the agent only uses its direct perception
            # and may not use information it gathered later during the trial 
        self.use_backward_pass_to_learn_d = use_backward_pass_to_learn_d



        self.assume_state_space_structure = state_structure_assumption
            # Either a list of AssumedSpaceStructure
            # or a signle AssumedSpaceStructure
            # Default : NO_STRUCTURE
            
        self.generalize_fadeout_function = (lambda x,param: np.exp(-max(param,0)*x))
        
        self.generalize_fadeout_function_temperature = 0.0
            # Generalization occurs relative to the position of the
            # actual inference

        self.learn_a = False
        self.learn_b = False
        self.learn_c = False
        self.learn_d = False
        self.learn_e = False
    
    def get_generalize_fadeout_function(self):
        return (lambda x : self.generalize_fadeout_function(x,self.generalize_fadeout_function_temperature))
