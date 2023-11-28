import numpy as np

from ..base.miscellaneous_toolbox import isField,listify,flexible_copy
from ..base.function_toolbox import normalize
from .utils import check_prompt_shape,reorder_axes


class link_function:
    """
    A small class to allow deepcopies of output-input connections.
    The user must specify what are the (layer) parameters. 
    Then specify a static function. e.g. (lambda a,b,c : a.o*b.s + b.s[:,99] + spm_kron(c.o,a.s))
    Here, a = layer 1 , b = layer 2, etc.

    Example of usage , with more or less complex function of the source layers
    model.inputs.o = link_function(process,(lambda x: x.o[0]))
    model.inputs.o = link_function([process,model_lower],function) 
                                        function = spm_kron(a,b)

    """
    def __init__(self,from_layers,static_function):
        self.static_function = static_function # MUST BE STATIC
        self.from_layers = listify(from_layers)
        self.from_layer_outputs = [lay.outputs for lay in self.from_layers]
        self.to_layer = None

    def sever(self):
        self.static_function = None
        self.from_layers = None
        self.from_layer_outputs = None
        self.to_layer = None
        
    def get(self):
        return self.static_function(*self.from_layer_outputs)



class layer_input : 
    def __init__(self,parentpointer):
        self.parent = parentpointer
        self.initialize_inputs(init_links=True)

    def initialize_inputs(self,init_links=True):
        # link_function, used by the layer to fetch data from other objects
        if (init_links):
            # Must all be link_functions !
            self.o = None   # Sequence of fixed observations
            self.s = None   # Sequence of fixed states 
            self.u = None   # Sequence of fixed actions

            self.o_d = None   # Distribution of observations
            self.s_d = None   # Distribution of states 
            self.u_d = None   # Distribution of actions

        # Actual values stored
        self.val_o = None   # Sequence of fixed observations
        self.val_s = None   # Sequence of fixed states 
        self.val_u = None   # Sequence of fixed actions

        self.val_o_d = None   # Distribution of observations
        self.val_s_d = None   # Distribution of states 
        self.val_u_d = None   # Distribution of actions

    def all_from_layers(self):
        def get_attr_connected_layers(attribute):
            if isField(attribute):
                # attribute.to_layer = self (needed ?)
                return attribute.from_layers
            return []
        
        all_connected_layers = [] # A list of all layers connected to this input
        all_connected_layers += get_attr_connected_layers(self.o)
        all_connected_layers += get_attr_connected_layers(self.s)
        all_connected_layers += get_attr_connected_layers(self.u)
        all_connected_layers += get_attr_connected_layers(self.o_d)
        all_connected_layers += get_attr_connected_layers(self.s_d)
        all_connected_layers += get_attr_connected_layers(self.u_d)
        return all_connected_layers

    def fetch(self):
        """ Get data from connected objects."""        
        def get_data(fetch_function):
            if isField(fetch_function):
                copied_val = flexible_copy(fetch_function.get())
                return copied_val
            return None

        self.val_o = get_data(self.o)   # Sequence of fixed observations
        self.val_s = get_data(self.s)   # Sequence of fixed states 
        self.val_u = get_data(self.u)   # Sequence of fixed actions

        self.val_o_d = get_data(self.o_d)   # Distribution of observations
        self.val_s_d = get_data(self.s_d)   # Distribution of states 
        self.val_u_d = get_data(self.u_d)   # Posterior policy distribution
    
    def clearMemory(self):
        # But not the link_functions >:(
        self.val_o = None   # Sequence of fixed observations
        self.val_s = None   # Sequence of fixed states 
        self.val_u = None   # Sequence of fixed actions

        self.val_o_d = None   # Distribution of observations
        self.val_s_d = None   # Distribution of states 
        self.val_u_d = None   # Distribution of actions

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
    
    def is_fixed_memory_empty(self):
        observation_input = isField(self.val_o)
        state_input = isField(self.val_s)
        action_input = isField(self.val_u)
        return not(observation_input or state_input or action_input)

    def is_dist_memory_empty(self):
        observation_input = isField(self.val_o_d)
        state_input = isField(self.val_s_d)
        action_input = isField(self.val_u_d)
        return not(observation_input or state_input or action_input)

    def is_input_memory_empty(self):
        return (self.is_fixed_memory_empty() and self.is_dist_memory_empty())
    
    def __str__(self):
        string_val = ""#"####################################################\n"
        string_val += ">> Input of layer " + self.parent.name+ " : \n"
        if (isField(self.val_o)):
            string_val +=" - o : \n"
            string_val += str(np.round(self.val_o,2))
            string_val += "\n----------- \n"
        if (isField(self.val_u)):
            string_val +=" - u : \n"
            string_val += str(np.round(self.val_u,2))
            string_val += "\n----------- \n"
        if (isField(self.val_s)):
            string_val +=" - s : \n"
            string_val += str(np.round(self.val_s,2))
            string_val += "\n----------- \n"
        if (isField(self.val_o_d)):
            string_val +=" - o_d : \n"
            string_val += str(np.round(self.val_o_d,2))
            string_val += "\n----------- \n"
        if (isField(self.val_u_d)):
            string_val +=" - u_d : \n"
            string_val += str(np.round(self.val_u_d,2))
            string_val += "\n----------- \n"
        if (isField(self.val_s_d)):
            string_val +=" - s_d : \n"
            string_val += str(np.round(self.val_s_d,2))
            string_val += "\n----------- \n"  
        if (self.is_no_input()):
            string_val += "NO PREDEFINED LINKS\n"
        return string_val

class layer_output : 
    def __init__(self,parentpointer):
        self.parent = parentpointer

        self.o = None   # Sequence of observed outcomes [0,...,T-1]
        self.u = None   # Sequence of selected actions [0,...,T-2]
        self.s = None   # Sequence of selected states [0,...,T-1]

        self.u_d = None  # Sequence of infered action distributions [0,...,T-2]
        self.s_d = None  # Sequence of infered states distributions [0,...,T-1]
        self.o_d = None  # Sequence of infered observation distributions [0,...,T-1]

    def clearMemory(self):

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
