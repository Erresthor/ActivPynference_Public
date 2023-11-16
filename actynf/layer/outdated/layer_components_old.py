import numpy as np

from ..base.miscellaneous_toolbox import isField,flexible_copy
from ..base.function_toolbox import normalize
from .utils import check_prompt_shape,reorder_axes

class layer_input : 
    def __init__(self,parentpointer):
        self.parent = parentpointer
        self.links = [] # pointers from outputs
        self.initialize_inputs()

    def initialize_inputs(self):
        # TEMPORARY PLACEHOLDERS WHEN RECEIVING DATA
        # When the input receives data from layerWires of different layerLinks,
        # it stores them here. If the input list 
        self.temp_o = []   # Sequence of fixed observations
        self.temp_s = []   # Sequence of fixed states 
        self.temp_u = []   # Sequence of fixed actions

        self.temp_o_d = []   # Distribution of observations
        self.temp_s_d = []   # Distribution of states 
        self.temp_u_d = []   # Distribution of actions

        # USED BY THE LAYER OBJECT
        self.o = None   # Sequence of fixed observations
        self.s = None   # Sequence of fixed states 
        self.u = None   # Sequence of fixed actions

        self.o_d = None   # Distribution of observations
        self.s_d = None   # Distribution of states 
        self.u_d = None   # Distribution of actions

    # Used to update part / all of the input variables : input Nmod <= input variable Nmod
    def update_definite_input(self,key,along_axes,value):
        to_value = getattr(self,key)
        if (not isField(to_value)):
            total_input_shape = check_prompt_shape(key,self.parent)
            definite_input_size = (len(total_input_shape),)
            to_value = np.full(definite_input_size,-1,dtype=int)
        if not(isField(along_axes)):
            along_axes = tuple(range(to_value.ndim))

        updated_to_value = flexible_copy(to_value)
        for k in range(len(along_axes)):
            updated_to_value[along_axes[k]] = value[k]
        setattr(self, key, updated_to_value)

    def update_distribution_input(self,key,along_axes,value):
        # I receive a distribution of ndim = len(along_axes)
        
        # In case along_axes is not sorted :
        value = reorder_axes(value,along_axes)

        to_value = getattr(self,key)
        if (not isField(to_value)):
            total_input_shape = check_prompt_shape(key,self.parent)
            to_value = normalize(np.ones(total_input_shape),all_axes=True)
        if not(isField(along_axes)):
            along_axes = tuple(range(to_value.shape[0]))

        to_value_shape = to_value.shape
        shape_of_new_input = list(to_value_shape)
        for i in range(len(to_value_shape)):
            if not(i in along_axes):
                shape_of_new_input[i] = 1
        reshape_of_new_input = tuple(shape_of_new_input)

        reshaped_from = np.reshape(value,reshape_of_new_input)
        reshaped_to = np.sum(to_value,along_axes,keepdims=True)

        updated_to_value = reshaped_from*reshaped_to
        assert (np.sum(updated_to_value)-1.0)<1e-10, "Error when updating the input for " + self.parent.name + " with the output of " + key + ". The final distribution sums to " + str(np.sum(updated_to_value)) + " instead of 1."
        
        setattr(self, key, updated_to_value)
    
    # Used to compress  : input Nmod <= input variable Nmod
    # def regularize_inputs(self):

    def is_fixed_input_empty(self):
        observation_input = isField(self.o)
        state_input = isField(self.s)
        action_input = isField(self.u)
        return not(observation_input or state_input or action_input)

    def is_dist_input_empty(self):
        observation_input = isField(self.o_d)
        state_input = isField(self.s_d)
        action_input = isField(self.u_d)
        return not(observation_input or state_input or action_input)

    def is_no_input(self):
        return (self.is_fixed_input_empty() and self.is_dist_input_empty())
    
    def __str__(self):
        string_val = ""#"####################################################\n"
        string_val += ">> Input of layer " + self.parent.name+ " : \n"
        if (isField(self.o)):
            string_val +=" - o : \n"
            string_val += str(np.round(self.o,2))
            string_val += "\n----------- \n"
        if (isField(self.u)):
            string_val +=" - u : \n"
            string_val += str(np.round(self.u,2))
            string_val += "\n----------- \n"
        if (isField(self.s)):
            string_val +=" - s : \n"
            string_val += str(np.round(self.s,2))
            string_val += "\n----------- \n"
        if (isField(self.o_d)):
            string_val +=" - o_d : \n"
            string_val += str(np.round(self.o_d,2))
            string_val += "\n----------- \n"
        if (isField(self.u_d)):
            string_val +=" - u_d : \n"
            string_val += str(np.round(self.u_d,2))
            string_val += "\n----------- \n"
        if (isField(self.s_d)):
            string_val +=" - s_d : \n"
            string_val += str(np.round(self.s_d,2))
            string_val += "\n----------- \n"  
        if (self.is_no_input()):
            string_val += "EMPTY\n"
        return string_val

class layer_output : 
    def __init__(self,parentpointer):
        self.parent = parentpointer
        self.links = [] # pointers to layerLinks

        self.o = None   # Sequence of observed outcomes [0,...,T-1]
        self.u = None   # Sequence of selected actions [0,...,T-2]
        self.s = None   # Sequence of selected states [0,...,T-1]

        self.u_d = None  # Sequence of infered action distributions [0,...,T-2]
        self.s_d = None  # Sequence of infered states distributions [0,...,T-1]
        self.o_d = None  # Sequence of infered observation distributions [0,...,T-1]

    def generate_outputs_from_STM(self,t,T,
                    o=None,o_d=None,
                    x=None,x_d=None,
                    u=None,u_d=None):
        """ 
        Take things from the STM and put them into the output
        """
        if (isField(o)):
            self.o = o[:,t]
        if (isField(x)):
            self.s = x[:,t]
        if (isField(u))and(t<T-1):
            self.u = np.array([u[t]]) # To allow links to be generic
        if (isField(o_d)):
            self.o_d = o_d[...,t]
        if (isField(x_d)):
            self.s_d = x_d[...,t]
        if (isField(u_d)and(t<T-1)):
            self.u_d = u_d[:,t]

    def is_fixed_output_empty(self):
        observation_input = isField(self.o)
        state_input = isField(self.s)
        action_input = isField(self.u)
        return not(observation_input or state_input or action_input)

    def is_dist_output_empty(self):
        observation_input = isField(self.o_d)
        state_input = isField(self.s_d)
        action_input = isField(self.u_d)
        return not(observation_input or state_input or action_input)

    def is_no_output(self):
        return (self.is_fixed_output_empty() and self.is_dist_output_empty())
    
    def __str__(self):
        string_val = ""#"####################################################\n"
        string_val += ">> Output of layer " + self.parent.name + " : \n"
        if (isField(self.o)):
            string_val +=" - o : \n"
            string_val += str(np.round(self.o,2))
            string_val += "\n----------- \n"
        print(self.u)
        if (isField(self.u)):
            string_val +=" - u : \n"
            string_val += str(np.round(self.u,2))
            string_val += "\n----------- \n"
        if (isField(self.s)):
            string_val +=" - s : \n"
            string_val += str(np.round(self.s,2))
            string_val += "\n----------- \n"
        if (isField(self.o_d)):
            string_val +=" - o_d : \n"
            string_val += str(np.round(self.o_d,2))
            string_val += "\n----------- \n"
        if (isField(self.u_d)):
            string_val +=" - u_d : \n"
            string_val += str(np.round(self.u_d,2))
            string_val += "\n----------- \n"
        if (isField(self.s_d)):
            string_val +=" - s_d : \n"
            string_val += str(np.round(self.s_d,2))
            string_val += "\n----------- \n"
        if (self.is_no_output()):
            string_val += "EMPTY\n"
        return string_val
