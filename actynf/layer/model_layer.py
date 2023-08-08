import numpy as np
from enum import Enum
import random
import time

from ..base.miscellaneous_toolbox import isField,listify,flexible_copy
from ..base.function_toolbox import normalize , spm_kron, spm_wnorm, nat_log , spm_psi, softmax,spm_dekron
from ..base.function_toolbox import sample_distribution,spm_complete_margin
from ..base.miscellaneous_toolbox  import flatten_last_n_dimensions

from .parameters.hyperparameters import hyperparameters
from .parameters.learning_parameters import learning_parameters
from .spm_forwards import spm_forwards
from .layer_learn import learn_from_experience
from .policy_tree import policy_tree_node

from .utils import dist_from_definite_outcome,minus1_in_arr
from .layer_link import layerLink
from .layer_components import layer_output,layer_input

EPSILON = 1e-12

class layerMode(Enum):
    PROCESS = 1
    MODEL = 2

class layer_variables :
    def __init__(self,layer):
        # Likelihood model a / A
        a_norm = normalize(layer.a)   # <=> A{m,g}
        a_prior = []             # <=> pA{m,g}
        a_novelty = []          # <=> W{m,g}
        a_ambiguity = []          # <=> H{m,g}
        a_kron = []
        a_kron_novelty = []
        a_kron_ambiguity = []
        for modality in range(layer.Nmod):
            a_prior.append(flexible_copy(layer.a[modality]))
            a_novelty.append(spm_wnorm(a_prior[modality])*(a_prior[modality]>0) )
            a_ambiguity.append(np.sum(a_norm[modality]*nat_log(a_norm[modality]),0).flatten())
            a_kron.append(flatten_last_n_dimensions(a_norm[modality].ndim-1,a_norm[modality]))
            a_kron_novelty.append(flatten_last_n_dimensions(a_novelty[modality].ndim-1,a_novelty[modality]))
            # a_kron_ambiguity.append(a_ambiguity[modality].flatten())
            # TODO : Check the calculations here

        # Transition model b / B
        b_norm = normalize(layer.b)
        b_prior = []
        b_complexity = []
        b_kron = []            # Kronecker form of policies
        b_complex_kron = []
        for factor in range(layer.Nf):   # For all factors
            b_prior.append(flexible_copy(layer.b[factor]))
            b_complexity.append(spm_wnorm(b_prior[factor])*(b_prior[factor]>EPSILON))

        # Some way of "compressing" multiple factors into a single matrix 
        # Slightly different from Matlab script, because our kronecker product orders dimension differently
        for k in range(layer.Np) :
            b_kron.append(1)
            b_complex_kron.append(1)
            for f in range(layer.Nf):
                b_kron[k] = spm_kron(b_kron[k],b_norm[f][:,:,layer.U[k,f]])
                b_complex_kron[k] = spm_kron(b_complex_kron[k],b_complexity[f][:,:,layer.U[k,f]])                

        # prior over initial states d/D
        d_norm = normalize(layer.d)
        d_prior = []
        d_complexity = []
        for f in range(layer.Nf):
            # Account for when we start with zero matrix
            if (np.sum(layer.d[f])<EPSILON) :
                layer.d[f] += EPSILON
            d_prior.append(flexible_copy(layer.d[f]))
            d_complexity.append(spm_wnorm(d_prior[f]))
        
        # Habit E
        # Account for when we start with zero matrix
        if (np.sum(layer.e)<EPSILON) :
            layer.e += EPSILON
        e_prior = np.copy(layer.e)
        e_log = nat_log(layer.e/np.sum(layer.e))
        
        # Preferences C
        c_base = []
        c_prior = []
        c_transformed = []
        for modality in range(layer.Nmod):
            c_transformed.append(spm_psi(layer.c[modality] + 1./32))
            c_prior.append(flexible_copy(layer.c[modality]))
            c_base.append(flexible_copy(layer.c[modality]))
        # Assure time invariance 
        for modality in range(layer.Nmod):
            if (layer.c[modality].shape[1] == 1) :
                c_base[modality] = np.tile(c_base[modality],(1,layer.T))
                c_transformed[modality] = np.tile(c_transformed[modality],(1,layer.T))
                c_prior[modality] = np.tile(c_prior[modality],(1,layer.T))
            c_base[modality] = nat_log(softmax(c_base[modality],0))
            c_transformed[modality] = nat_log(softmax(c_transformed[modality],0))
        c = c_transformed if(layer.learn_options.learn_c) else c_base
        
        self.a = a_norm
        self.a_prior =  a_prior
        self.a_novelty = a_novelty
        self.a_ambiguity = a_ambiguity
        self.a_kron = a_kron
        self.a_kron_novelty = a_kron_novelty
        
        self.b = b_norm
        self.b_prior = b_prior
        self.b_complexity = b_complexity
        self.b_kron = b_kron
        self.b_kron_complexity = b_complex_kron
            
        self.c = c
        self.c_prior = c_prior

        self.d = d_norm
        self.d_prior = d_prior
        self.d_complexity = d_complexity

        self.e= e_log
        # print(e_log.shape)
        self.e_prior= e_prior
    
class layer_STM :
    def __init__(self,Nmod,No,Nf,Ns,Np,T):
        self.o_d = np.full(tuple(No)+(T,),-1.0) # Observation DISTRIBUTIONS
        self.x_d = np.full(tuple(Ns)+(T,),-1.0) # State DISTRIBUTIONS
        self.u_d = np.full((Np,)+(T-1,),-1.0) # Action DISTRIBUTIONS
        self.o = np.full((Nmod,)+(T,),-1) # Observation 
        self.x = np.full((Nf,)+(T,),-1) # State 
        self.u = np.full((T-1,),-1) # 1dimensionnal : which action did I pick last time ? 
            # (one action may be comprised of several state transitions )
            # ==> one can access state transitions from action by using self.U :
            # picked_state_transition(at a given time t across all factors) = self.U[self.u[t],:]
        # self.x_kron = []
        # for t in range(T):
        #     self.x_kron.append(spm_kron(d))
    
    def is_value_exists(self,key,t):
        if (key=="o"):
            return not(minus1_in_arr(self.o[:,t]))
        elif (key=="o_d"):
            return not(minus1_in_arr(self.o_d[...,t]))
        elif (key=="x"):
            return not(minus1_in_arr(self.x[:,t]))
        elif (key=="x_d"):
            return not(minus1_in_arr(self.x_d[...,t]))
        elif (key=="u"):
            return not(minus1_in_arr(self.u[t]))
        elif (key=="u_d"):
            return not(minus1_in_arr(self.u_d[:,t]))

    def __str__(self):
        return self.getString()

    def getString(self,t=None):
        string_val = "_________________________________________________\n"
        string_val += "STM [Short Term Memory] : \n"
        if (t==None):
            string_val += "o : \n"
            string_val += str(np.round(self.o,2))
            string_val += "\n----------- \n"
            string_val += "o_d : \n"
            string_val += str(np.round(self.o_d,2))
            string_val += "\n----------- \n"
            string_val += "x : \n"
            string_val += str(np.round(self.x,2))
            string_val += "\n----------- \n"
            string_val += "x_d : \n"
            string_val += str(np.round(self.x_d,2))
            string_val += "\n----------- \n"
            string_val += "u : \n"
            string_val += str(np.round(self.u,2))
            string_val += "\n----------- \n"
            string_val += "u_d : \n"
            string_val += str(np.round(self.u_d,2))
            string_val += "\n----------- \n"
        else :
            string_val += "(AT TIME " + str(t) +")\n"
            string_val += "o : \n"
            string_val += str(np.round(self.o[:,t],2))
            string_val += "\n----------- \n"
            string_val += "o_d : \n"
            string_val += str(np.round(self.o_d[...,t],2))
            string_val += "\n----------- \n"
            string_val += "x : \n"
            string_val += str(np.round(self.x[:,t],2))
            string_val += "\n----------- \n"
            string_val += "x_d : \n"
            string_val += str(np.round(self.x_d[...,t],2))
            string_val += "\n----------- \n"
            string_val += "u : \n"
            try :
                string_val += str(np.round(self.u[t],2))
            except :
                string_val += "// Over temporal horizon // \n"
            string_val += "\n----------- \n"
            string_val += "u_d : \n"
            try :
                string_val += str(np.round(self.u_d[:,t],2))
            except :
                string_val += "// Over temporal horizon // \n"
            string_val += "\n----------- \n"
        string_val += "_________________________________________________\n"
        return string_val

class mdp_layer :
    """ 
    I am a MDP layer. I operate independently.
    This is a very generic class with generic components, which can be used both for generative process AND model.
    
    observations ----> [ layer ] ----> q_s and q_pi

    action + state ---> [ layer ] ---> observation / observation distr

    notable functions : 
    - layer learn : use the in-memory observations and actions of the last trial to update the current model
    - layer update : update the current beliefs about action and states given a new perception stimuli
    """
    
    # NOTE : a layer is defined by a single action modality.

    # Agent constructor
    def __init__(self,name,mode="model",
                 A = None,B=None,C=None,D=None,E=None,
                 inU = None,
                 T = None,T_horiz=2,
                 in_seed = None):
        self.name = name
        self.verbose = False
        self.debug = False

        # Seeding
        if (not(isField(in_seed))):
            in_seed = random.randint(0,9999)
        self.seed = in_seed
        self.RNG = None
        self.reseed()

        assert (mode == "model" or mode=="process"),"The layer should either be a process or a model ! (currently " + str(mode) + ")"
        self.layerMode = (layerMode.MODEL if mode=="model" else layerMode.PROCESS)

        # self.update_frequency = 1 # ]0,1] 
                # --> At most, you can be updated once every loop, at worst, only once in total$
                # Could be a probability of being selected every loop, describe cognitive processes at different timescales
                # To be implemented
        
        # Layer simulation parameters 
        self.t = 0          # The current time step, if t==T, the experience is over  
        self.T = T          # The final time step according to the agent
        self.T_horizon = T_horiz

        # Layer parameters
        self.hyperparams = hyperparameters()
        self.learn_options = learning_parameters()

        #Model building blocks -----------------------------------------------------------------------------------------------
        # Beliefs (learning process, =/= generative process) --> What our agent believes about the dynamics of the world
        self.a = flexible_copy(A)       # A matrix 
        self.b = flexible_copy(B)       # B matrix 
        self.c = flexible_copy(C)       # C matrix
        self.d = flexible_copy(D)       # D matrix
        self.e = flexible_copy(E)       # E matrix

        self.U = flexible_copy(inU)       # Perceived space of actions
        self.V = None       # Perceived space of policies
        
        self.var = None   # Once primed, the functions above will be stocked here to run the calculations

        # Things will go into the layer and get out of it
        self.inputs = layer_input(self)
        self.outputs = layer_output(self)

        # Class properties : 
        self.Nmod = None
        self.No = None
        self.Nf = None
        self.Ns = None
        self.Nu = None
        self.Np = None

        # STM : Short term memory for the layer : 
        self.STM = None
        # CONTAINS THE FOLLOWING :
        # self.o_d = None # Observation DISTRIBUTIONS
        # self.x_d = None # State DISTRIBUTIONS
        # self.u_d = None # Action DISTRIBUTIONS
        # self.o = None # Observation 
        # self.x = None # State 
        # self.u = None # Action 
        # NOT self.x_kron = None # Kronecker distribution of posterior over states

        self.check() # Run a few checks & initialize property fields
        self.initialize_STM()

    def reseed(self,new_seed=None):
        if (not(isField(new_seed))):
            assert type(self.seed)==int,"No pre existing seed, please provide one ..."
        else : 
            self.seed = new_seed
        self.RNG = random.Random(self.seed)

    def copy(self,newName=None):
        if (isField(newName)):
            new_layer_name = newName
        else :
            new_layer_name = self.name+"_copy"

        new_layer =  mdp_layer(new_layer_name,self.a,self.b,self.c,self.d,self.e,
                         self.U,self.T,self.seed)
        
        
        new_layer.T = self.T
        new_layer.T_horizon = self.T_horizon

        # Copied layers share the same hyperparameters & learning parameters
        new_layer.hyperparams = self.hyperparams
        new_layer.learn_options = self.learn_options
        new_layer.layerMode = self.layerMode
        return new_layer
    
    def clear_inputs(self):
        self.inputs = layer_input(self)

    def clear_outputs(self):
        self.outputs = layer_output(self)

    def check(self):
        if(self.name == '') :
            self.name = 'unnamed_layer'

        pcmm = "" #potential_component_missing_message
        if not(isField(self.a)): pcmm += self.name +" : A not filled in " +  "\n"
        if not(isField(self.b)): pcmm += self.name +" : B not filled in " +  "\n"
        if not(isField(self.c)): pcmm += self.name +" : C not filled in " + "\n"
        if not(isField(self.d)): pcmm += self.name +" : D not filled in " + "\n"
        if not(isField(self.e)): pcmm += self.name +" : E not filled in " + "\n"
        if not(isField(self.U)): pcmm += self.name +" : U not filled in " + "\n"
        assert (pcmm==""),print("ERROR : \n" + pcmm +"\n")

        # Ensure the layer functions are lists
        self.a = listify(self.a)
        self.b = listify(self.b)
        self.c = listify(self.c)
        self.d = listify(self.d)
        self.e = self.e

        # Get the dimensions of the layer space
        self.Nmod = len(self.a)
        self.No = []
        for i in range(self.Nmod) :
            self.No.append(self.a[i].shape[0])


        self.Nf = len(self.d)
        self.Ns = []
        for i in range(self.Nf):
            self.Ns.append(self.d[i].shape[0])


        self.Np = self.U.shape[0] # Number of allowable set of actions
        if (self.U.ndim==1):
            assert len(self.b) == 1,"Error : action matrix for layer " + str(self.name) + " is 1-dimensional but transition matrix is " + str(len(self.b)) + "-dimensionnal."
            self.U = np.expand_dims(self.U,1)

        self.Nu = []  # Number of allowable actions for each factor
        for f in range(self.Nf) :
            assert self.b[f].ndim > 2,self.name + " : B has too little dimensions"
            self.Nu.append(self.b[f].shape[2])
        
        # TODO : add more checks (outcomes, states, etc)
        assert len(self.c)==self.Nmod, "The preference matrix number of modalities (" + str(len(self.c)) + ") should match the perception model number of modalities (" + str(len(self.a)) + ")."
        assert len(self.b)==self.Nf,"The action matrix number of factors (" + str(len(self.b)) + ") should match the initial state matrix number of factors (" + str(len(self.d)) + ")."
        for mod in range(self.Nmod):
            if (self.c[mod].ndim==1):
                assert self.c[mod].shape[0] == self.No[mod],"Error : preferences for modality " + str(mod) + " are 1dimensional and do no fit the number of potential outcomes ("+str(self.No[mod])+")."
                self.c[mod] = np.expand_dims(self.c[mod],1)

    def initialize_STM(self): 
        self.STM = layer_STM(self.Nmod,self.No,self.Nf,self.Ns,self.Np,self.T)

    # COSMETIC
    def getCurrentMatrices(self):
        report = "Layer weights :\n"
        report += "   Matrix a :\n"
        for mod in range(self.Nmod):
            report += "     Modality " + str(mod) + " :\n"
            report += str(np.round(self.a[mod],2)) + "\n"
        
        report += "   Matrix b :\n"
        for factor in range(self.Nf):
            for act in range(self.Nu[factor]):
                report += "     Factor : " + str(factor) + "  --- Transition " + str(act) + " :\n"
                report += str(np.round(self.b[factor][:,:,act],2))+ "\n"
        
        report += "   Matrix c :\n"
        for mod in range(self.Nmod):
            report += "Modality " + str(mod) + " :\n"
            report += str(np.round(self.c[mod],2))+ "\n"
        
        report += "   Matrix d :\n"
        for fac in range(self.Nf):
            report += "     Factor " + str(fac) + " :\n"
            report += str(np.round(self.d[fac],2))+ "\n"

        report += "   Matrix e :\n"
        report += str(np.round(self.e,2))+ "\n"

        report += "   Allowable actions u :\n"
        report += str(self.U)+ "\n"
        return report

    def get_dimension_report(self):
        report = "-------------------------------------\n"
        report += "LAYER DIMENSION REPORT ("+self.name+"): \n\n"
        report +=  "Observation modalities : "+ str(self.Nmod) + "\n"
        for mod in range(self.Nmod):
            report += "    Modality " + str(mod) + " : " + str(self.No[mod]) + " outcomes.\n"
        report += "Hidden states factors : "+ str(self.Nf) + "\n"
        for state in range(self.Nf):
            report += "    Model factor " + str(state) + " : " + str(self.Ns[state]) + " possible states. \n"
        report += "Number of potential actions : "+ str(self.Np) + "\n"
        for factor in range(self.Nf):
            report += "    Factor " + str(factor) + " : " + str(self.Nu[factor]) + " possible transitions. \n"
        report += "-------------------------------------\n"
        return report
    
    def __str__(self):
        return "LAYER " + self.name + " : \n " + self.get_dimension_report() + "\n##################################################\n" + self.getCurrentMatrices() + "##################################################"
      
    # Utilitaries
    def factorwise_action_model_average(self, action_distribution):
        return_this_matrix_list = []
        for factor in range(self.Nf):
            sum_of_matrices = 0
            for policy in range(self.Np):
                action_K_done = self.U[policy,factor]
                prob_action_K = action_distribution[policy]
                sum_of_matrices += prob_action_K*self.var.b[factor][:,:,action_K_done]
            return_this_matrix_list.append(normalize(sum_of_matrices))
        return return_this_matrix_list

    def kronecker_action_model_average(self, action_distribution):
        kron_b = self.var.b_kron
        total = 0
        for pol in range(len(kron_b)):
            total = total + kron_b[pol]*action_distribution[pol]
        return total

    def get_factorwise_actions(self,at_time=0):
        assert at_time<self.T-1,"Can't get kronecker form of hidden states at time " + at_time + " (Temporal horizon reached)"
        return (self.U[self.STM.u[at_time],:]).tolist()

    def get_kron_state_at_time(self,at_time=0):
        assert at_time<self.T,"Can't get kronecker form of hidden states at time " + at_time + " (Temporal horizon reached)"
        if (self.STM.is_value_exists("x_d",at_time)):
            return self.joint_to_kronecker(self.STM.x_d[...,at_time])
        else : 
            return spm_kron(self.var.d)

    def joint_to_kronecker(self,joint):
        """ Warning ! For a single timestep."""
        return joint.flatten('C')

    def joint_to_kronecker_accross_time(self,joint):
        """ Warning ! For a single timestep."""
        return joint.reshape(-1, joint.shape[-1]) # I don't like it but I'm too lazy to change this
    
    def kronecker_to_joint(self,x_kron):
        return np.reshape(x_kron,self.Ns,'C')
    
    def kronecker_to_joint_accross_time(self,x_kron):
        return np.reshape(x_kron,self.Ns + [-1],'C')

    def dekron_state(self,x_kron,at_time=0):
        assert at_time<self.T,"Can't get kronecker form of hidden states at time " + at_time + " (Temporal horizon reached)"
        return  spm_dekron(x_kron[at_time],tuple(self.Ns))
    
    def get_total_number_of_hidden_states(self):
        total = 1
        for k in range(len(self.Ns)):
            total = total*self.Ns[k]
        return total

    # Ticks
    def use_inputs_to_populate_STM(self):
        """ 
        Checks this layer's inputs. If not empty,
        updates this layer's STM with the elements.
        A distribution will always be available for the 
        filled modalities.
        """
        t = self.t 
        T = self.T

        if(self.inputs.is_no_input()):
            if (t==0):
                if (self.verbose):
                    print("No inputs to this layer. This ought to be the initial step of the generative process.")
            else : 
                raise ValueError("No valid inputs were detected for the layer at time "+str(t)+". The layer can't be updated.")
        
        if (isField(self.inputs.o_d)):
            assert self.STM.o_d[...,t].shape == self.inputs.o_d.shape ,"Observation dist input size " + str(self.inputs.o_d.shape) + " should fit layer awaited outcome size " + str(self.STM.o_d[...,t].shape) + " ."
            self.STM.o_d[...,t] = self.inputs.o_d
        if (isField(self.inputs.s_d)):
            assert self.STM.x_d[...,t].shape == self.inputs.s_d.shape ,"State dist input size " + str(self.inputs.s_d.shape) + " should fit layer awaited state size " + str(self.STM.x_d[...,t].shape) + " ."
            self.STM.x_d[...,t] = self.inputs.s_d
        if (isField(self.inputs.u_d)):
            assert self.STM.u_d[:,t].shape == self.inputs.u_d.shape ,"Action dist input size " + str(self.inputs.u_d.shape) + " should fit layer awaited action dist size " + str(self.STM.u_d[:,t].shape) + " ."
            self.STM.u_d[:,t] = self.inputs.u_d

        if (isField(self.inputs.o)):
            assert self.inputs.o.shape == self.STM.o[:,t].shape ,"Observation input size " + str(self.inputs.o.shape) + " should fit layer awaited modality size " + str(self.STM.o[:,t].shape) + " ."
            self.STM.o[:,t] = self.inputs.o

            # If o is an input for the layer AND
            # If o_d does not exist in the stm at time t
            # Then o_d is a distribution with value 1 at
            # the observed index and 0 elsewhere.
            if not(self.STM.is_value_exists("o_d",t)):
                O,list_O = dist_from_definite_outcome(self.inputs.o,self.No)
                self.STM.o_d[...,t] = O
        
        if (isField(self.inputs.s)):
            assert self.inputs.s.shape == self.STM.x[:,t].shape ,"State input size " + str(self.inputs.s.shape) + " should fit layer awaited factor size " + str(self.STM.x[:,t].shape) + " ."
            self.STM.x[:,t] = self.inputs.s
            
            # If s is an input for the layer AND
            # If s_d does not exist in the stm at time t
            # Then s_d is a distribution with value 1 at
            # the observed index and 0 elsewhere.
            if not(self.STM.is_value_exists("x_d",t)):
                
                S,list_S = dist_from_definite_outcome(self.inputs.s,self.Ns)               
                self.STM.x_d[...,t] = S


        if (t>0):
            # Action inputs can only change previous timesteps
            # Before the last timestep
            if (isField(self.inputs.u)):
                assert self.inputs.u.shape == (1,) ,"Action input size " + str(self.inputs.u.shape) + " should be (1,)."
                self.STM.u[t-1] = self.inputs.u

                # If u is an input for the layer AND
                # If u_d does not exist in the stm at time t
                # Then u_d is a distribution with value 1 at
                # the observed index and 0 elsewhere.
                if not(self.STM.is_value_exists("u_d",t-1)):
                    U,list_U = dist_from_definite_outcome(self.inputs.u,[self.Np])
                    self.STM.u_d[:,t-1] = U

    # Full trial mechanics
    def prime_model_functions(self):
        # Use inputs to initialize blocks :
        self.var = layer_variables(self)

    # GENERATIVE PROCESS :
    def generate_observations(self):
        """ 
        Uses true states & previous actions contained in memory
        to populate the distribution of states, if none is provided.
        Then, if no observation is provided, it uses the current state
        to populate the distribution of the observations in the memory.
        TODO : implement a version of this which :
            - Allows to fix some values in the observation / state map but allowing other data
            to be generated MDP style
        """

        # Inputs needed to generate observations: 
        #IF (NO o_d) :
        #   EITHER : 
        #       - x_d[t] (the current state distribution)
        #       - |if t>0 :x_d[t-1] and u_d[t-1] (the previous state distribution & previous action)
        #         |else : d
        #ELSE :
        #   - o_d (if we have this one, previous states & actions will not be computed by the layer
        #           thus using this once will prevent using the MDP part of the generative process
        #           for the remainder of the trial (ouch))
        t = self.t

        if (self.STM.is_value_exists("o_d",t)):
            # The new observations are preset in memory 
            o_d = self.STM.o_d[...,t]
            if (self.STM.is_value_exists("o",t)):
                o = self.STM.o[:,t].astype(np.int)
            else : 
                o = sample_distribution(o_d,random_number_generator=self.RNG) # This is a tuple
                o = np.asarray(o).astype(np.int)
                # outputArray = np.asarray(inputTuple)
            x = self.STM.x[:,t]         # Input states stay the same
            x_d = self.STM.x_d[...,t]   # 
        else :
            if (self.STM.is_value_exists("x_d",t)):
                # The new states are preset in memory
                x_d = self.STM.x_d[...,t]
                if (self.STM.is_value_exists("x",t)):
                    x = self.STM.x[:,t].astype(np.int)
                else : 
                    x = sample_distribution(x_d,random_number_generator=self.RNG) # This is a tuple
                    x = np.asarray(x).astype(np.int)
            else : 
                # Nothing preset in memory ! 
                # We should use the data stocked in the STM 
                # to infer the new states & observations
                if (t==0):
                    x_d_kron = spm_kron(self.var.d)
                else : 
                    if not(self.STM.is_value_exists("x_d",t-1)):
                        raise ValueError("No previous states were found in the layer's STM for t>0.")
                    if not(self.STM.is_value_exists("u_d",t-1)):
                        raise ValueError("No previous actions were found in the layer's STM for t>0.")
                    

                    # Previous states can either be a probability distribution or a definite quantity. We get them from x_d and u_d either way :
                    x_d_tprevious = self.STM.x_d[...,t-1]
                    u_d_tprevious = self.STM.u_d[...,t-1]

                    # Factor by factor
                    # transition_tprevious = self.action_model_average(u_d_tprevious)

                    # Joint form of factors & transitions : 
                    kron_transition_tprevious = self.kronecker_action_model_average(u_d_tprevious)
                    kron_form_x_d_tprevious = self.joint_to_kronecker(x_d_tprevious)

                    x_d_kron = np.dot(kron_transition_tprevious,kron_form_x_d_tprevious)
                x_d = self.kronecker_to_joint(x_d_kron)
                x = np.asarray(sample_distribution(x_d,random_number_generator=self.RNG)).astype(np.int)
            # x and x_d are available : let's use them and a to 
            # generate the corresponding 
            o = np.zeros((self.Nmod,))

            generate_outcome = "from_sampled_state"
            generate_outcome = "from_distribution"

            po_list = []
            # Probabilistic outcome distribution (from x_d)
            if (generate_outcome == "from_distribution"):
                for modality in range (self.Nmod):
                    # a_matrix = self.var.a[modality]
                    # flattened_a = flatten_last_n_dimensions(a_matrix.ndim-1,a_matrix)
                    flattened_a_mat = self.var.a_kron[modality]
                    qx = self.joint_to_kronecker(x_d)
                    o_d_mod = np.dot(flattened_a_mat,qx)
                    o[modality] = sample_distribution(o_d_mod,random_number_generator=self.RNG)[0]
                    po_list.append(o_d_mod)
                o = o.astype(np.int)

            # Deterministic from the sampled state (x) :
            if (generate_outcome=="from_sampled_state"):
                for modality in range(self.Nmod):
                    ind = (slice(None),) + tuple(x[:])  # Index corresponding to the current active states
                    o_d_mod = self.var.a[modality][ind]
                    o[modality] = sample_distribution(o_d_mod,random_number_generator=self.RNG)[0]
                    po_list.append(o_d_mod)
                o = o.astype(np.int)
            
            # joint distribution : 
            o_d = np.reshape(spm_kron(po_list),self.No)
        # x, x_d, o and o_d are available 
        # Warning, some may be "None"
        return o,o_d,x,x_d

    def process_update(self,use_definite_distribution=True):
        t = self.t
        o,o_d,x,x_d = self.generate_observations()
        # o is the whole task's observations
        # o_d is the distribution from  which o
        # is sampled, just for this timestep
        # Should it be changed ?
        self.STM.o[:,t] = o
        self.STM.o_d[...,t] = o_d
        self.STM.x[:,t] = x
        if (use_definite_distribution):
            self.STM.x_d[...,t] = dist_from_definite_outcome(x,self.Ns)[0]
        else :
            self.STM.x_d[...,t] = x_d
        
    def process_tick(self,
                update_t_when_over=True,clear_inputs_when_over=True):
        t = self.t

        self.use_inputs_to_populate_STM()
        self.process_update()
        self.outputs.generate_outputs_from_STM(t,self.T,
            o = self.STM.o,o_d=self.STM.o_d, # o, o_d
            x=self.STM.x,x_d=self.STM.x_d) # x, x_d
        
        if (update_t_when_over):
            self.t = t+1
        if(clear_inputs_when_over):
            print("Clearing inputs !")
            self.clear_inputs()

    # GENERATIVE MODEL :

    def belief_propagation(self,verbose=False):
        """ 
        Use prior beliefs about :
         - States
         - Policies
         + previous observations 
        stored in the STM.
        ==> to infer which states are the most likely and pick 
        prompt a policy tree search to select the actions that 
        maximize self evidence.
        """
        t0 = time.time()
        t = self.t
        if (self.STM.is_value_exists("o_d",t)):
            list_O = spm_complete_margin(self.STM.o_d[...,t])
        else : 
            raise ValueError("No observations were found in the layer's STM for t = "+str(t)+". \n Found o = \n" + str(np.round(self.STM.o_d[...,t],2)))

        # Fetch priors over subsequent states
        if (t>0):
            Q_t_previous = self.joint_to_kronecker(self.STM.x_d[...,t-1])
            last_action = self.STM.u[t-1]
            last_transition = self.var.b_kron[last_action]
            Q_t = np.dot(last_transition,Q_t_previous)
        else :
            Q_t = spm_kron(self.var.d)
        P = flexible_copy(Q_t) # Prior over current states 
        if verbose :
            print("Prior Q --> " +str(np.round(Q_t,2)) + " -  Obs : " + str(list_O))
        tree = policy_tree_node(0,P,self.U.shape[0],self.get_total_number_of_hidden_states())

        forward_t = t
        
        # The maximum horizon is the last timestep : T-1
        # Else, it is just the current time + temporal horizon :
        # If t horizon = 0, N = t, no loop
        # If t horizon = 1, N = t+1,there is one instance of t<N leading to another recursive tree search loop
        # If t horizon = 2, N = t+2, there are two nested instances of recursive search
        G,Q  = spm_forwards(list_O,P,self.U,self.var,forward_t,
                    self.T,min(self.T-1,t+self.T_horizon),tree,self.debug)
        # DEBUG : 
        if (self.debug):
            print("Free energy at time " + str(t) + " :")
            print(G)
            if verbose : 
                print("Post Q --->" + str(np.round(Q,2)))
        return G,Q,time.time() - t0,tree

    def pick_action(self):
        """ 
        Use posterior beliefs about policy stored in the STM
        to pick which actions I should conduct at time t.
        """
        t = self.t 

        if (t>=self.T-1) :
            # No action selection needed here
            return 

        Ru = softmax(self.hyperparams.alpha * nat_log(self.STM.u_d[:,t]))
        action_selected_tuple = sample_distribution(Ru,random_number_generator=self.RNG)
        # print(Ru,action_selected_tuple[0])
        return action_selected_tuple,self.U[action_selected_tuple[0],:]
    
    def model_update(self):
        t = self.t
        G,Q,prop_time,tree = self.belief_propagation()
        
        # Update the STM with the inference results
        # Short term memory addition : 
        
        self.STM.x_d[...,t] = self.kronecker_to_joint(Q)
        if (t<self.T-1):
            # # posterior_over_policy & precision
            softmax_posterior_u = softmax(G)
            self.STM.u_d[:,t] = softmax_posterior_u
            w = np.inner(softmax_posterior_u,nat_log(softmax_posterior_u))

            # Pick an action & save the result to STM
            u_idx, state_u_idx = self.pick_action()
            self.STM.u[t] = u_idx[0]
        return tree

    def model_tick(self,
                   update_t_when_over=True,
                   clear_inputs_when_over=True):
        t = self.t
        self.use_inputs_to_populate_STM()
        searchtree = self.model_update()
        self.outputs.generate_outputs_from_STM(t,self.T,  # Outputs the results
                x_d = self.STM.x_d,  #qs
                u=self.STM.u,u_d = self.STM.u_d) #u, qu        
        
        if (update_t_when_over):
            self.t = t+1
        if(clear_inputs_when_over):
            self.clear_inputs()
        return searchtree

    def model_learn(self):
        """ 
        Do stuff :D
        """
        learn_from_experience(self)

    # GENERAL LAYER FUNCTIONS (CALLED BY NETWORK)

    def prerun(self,verbose=False):
        if (verbose):
            print("Priming " + self.name)
        self.prime_model_functions()
        self.initialize_STM()
        self.inputs.initialize_inputs()
        self.t = 0

    def tick(self,update_t_when_over=True,
                   clear_inputs_when_over=True):
        returns = None
        if self.layerMode == layerMode.MODEL:
            returns = self.model_tick(update_t_when_over,clear_inputs_when_over)
        elif self.layerMode == layerMode.PROCESS : 
            returns = self.process_tick(update_t_when_over,clear_inputs_when_over)
        return returns
    
    def postrun(self,verbose=False):
        if (verbose):
            print("Updating " + self.name +" 's beliefs")
        if self.layerMode == layerMode.MODEL:
            returns = self.model_learn()
        elif self.layerMode == layerMode.PROCESS : 
            pass

    def tick_generator(self):
        t = self.t
        while t<self.T :
            returns = self.tick(False,False)
            # returns = self.model_tick(False,True)
            yield returns
            self.t = self.t + 1
        return

    # Running the layer "manually"
    def full_run_generator(self):
        t = self.t
        self.prerun()
        while t<self.T :
            returns = self.tick(False,False)
            # returns = self.model_tick(False,True)
            yield returns
            self.t = self.t + 1
        self.postrun()
        return
        

    # LAYER LINK MECHANICS
    def get_links(self,key="from"):
        links = []
        if (key=="from"):
            for from_self_link in (self.outputs.links) :
                links.append(from_self_link)
        elif (key=="to"):
            for to_self_link in (self.inputs.links) :
                links.append(to_self_link)
        return links 
    
    def get_linked_layers(self):
        connected_layers_upstream = [] # upstream nodes --> self
        connected_layers_downstream =  []  # self --> dowstream nodes
        
        for from_self_link in (self.outputs.links) :
            connected_layers_upstream.append([from_self_link,from_self_link.to_input.parent])
        for to_self_link in (self.inputs.links) :
            connected_layers_downstream.append([to_self_link,to_self_link.from_output.parent])
        return {"to_self" : connected_layers_upstream,"from_self" :connected_layers_downstream}

    def get_connection_weighted_linked_layers(self):
        linkd_layers_dict = self.get_linked_layers()
        upstream_weighted = []
        downstream_weighted = []
        for upstream_layer in linkd_layers_dict["from_self"]:
            is_model_to_process = (upstream_layer[1].layerMode == layerMode.MODEL) and (self.layerMode == layerMode.PROCESS)
            upstream_weighted.append([upstream_layer[1],len(upstream_layer[0].connections),is_model_to_process])
        for downstream_layer in linkd_layers_dict["to_self"]:
            is_model_to_process = (downstream_layer[1].layerMode == layerMode.PROCESS) and (self.layerMode == layerMode.MODEL)
            downstream_weighted.append([downstream_layer[1],len(downstream_layer[0].connections),is_model_to_process])
        return {"to_self" : upstream_weighted,"from_self" : downstream_weighted}
    
    def transmit_outputs(self):
        for link in self.outputs.links :
            link.fire_all()
