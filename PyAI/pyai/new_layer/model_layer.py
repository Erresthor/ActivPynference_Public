from .parameters.hyperparameters import hyperparameters
from .parameters.learning_parameters import learning_parameters



class model_layer :
    """ 
    I am a layer of an agent generative model. I operate independently.
    This is a very generic class with generic components.
    
    observations ----> [ layer ] ----> q_s and q_pi

    notable functions : 
    - layer learn : use the in-memory observations and actions of the last trial to update the current model
    - layer update : update the current beliefs about action and states given a new perception stimuli
    """

    # Agent constructor
    def __init__(self,in_seed = 0):
        self.name = 'default name'
        self.verbose = False

        self.seed = in_seed

        self.hyperparams = hyperparameters()
        self.learn_options = learning_parameters()
        self.cache =  None

        #Model building blocks -----------------------------------------------------------------------------------------------
        # Beliefs (learning process, =/= generative process) --> What our agent believes about the dynamics of the world
        self.a = None       # A matrix beliefs (learning)
        self.b = None       # B matrix beliefs (learning)
        self.c = None       # C matrix beliefs (learning)
        self.d = None       # D matrix beliefs (learning)
        self.e = None       # E matrix beliefs (learning)
        
        self.U = None       # Perceived space of actions

        self.trial = 0      # Trial counter


        
        # Simulation parameters --> Different structure ?
        self.t = 0          # The current time step, if t==T, the experience is over  
        self.T = 0          # The temporal horizon for this layer (a.k.a, how many timescales it will have to experience)
              
        self.o = None   # Sequence of observed outcomes
        self.u = None   # Sequence of selected actions

        self.q_s = None # Sequence of infered states
        self.q_pi = None # Sequence of infered actions

    def layer_tick(self):

    
    def layer_learn(self,learning_function):
        learning_function(self)

