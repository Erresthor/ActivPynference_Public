from enum import Enum




class GenerativeProcessType(Enum):
    MDP = 0
    FIXED = 1
    MIXED = 2

class GenerativeProcess :
    # I return a generator making observations depending on actions and current time t

    def __init__(self,gen_proc_type):
        self.type = gen_proc_type

    
    def get_generator(self):
        if (self.type == GenerativeProcessType.MDP):
            
            
            
            def generate_observations(u,t):
                return u+t



            
        
            return generate_observations
        else : 
            print("Sorry, this generative process type has not been implemented yet.")



class mdp_generator :
    def __init__(self,A,B,C,D,E,o,s):
        # For now, the generative process are defined by their mdp components or overriding o or s sequences
