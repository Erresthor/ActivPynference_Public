# from demo.my_projects.internal_observator_hypothesis import get_nf_network,ind2sub,sub2ind
# from demo.my_projects.generic_neurofeedback.build import *
import numpy as np



def extract(a):
    return a.o

def multiply_obs(a,b):
    return a.o * b.o

class output :
    def __init__(self):
        self.o = None   # Sequence of observed outcomes [0,...,T-1]
        self.u = None   # Sequence of selected actions [0,...,T-2]
        self.s = None   # Sequence of selected states [0,...,T-1]

        self.u_d = None  # Sequence of infered action distributions [0,...,T-2]
        self.s_d = None  # Sequence of infered states distributions [0,...,T-1]
        self.o_d = None  # Sequence of infered observation distributions [0,...,T-1]

class fetch_function:
    def __init__(self,static_function,dyn_args):
        self.static_function = static_function # MUST BE STATIC
        self.my_args = dyn_args  # A list of arguments
        
    def get(self):
        return self.static_function(*self.my_args)

class input:
    def __init__(self, source=None):
        self.o_value = None   # Sequence of observed outcomes [0,...,T-1]
        self.u_value = None   # Sequence of selected actions [0,...,T-2]
        self.s_value = None   # Sequence of selected states [0,...,T-1]

        self.u_d_value = None  # Sequence of infered action distributions [0,...,T-2]
        self.s_d_value = None  # Sequence of infered states distributions [0,...,T-1]
        self.o_d_value = None  # Sequence of infered observation distributions [0,...,T-1]

        # FUNCTIONS THAT TAKE VALUES FROM FROMOBJECTS
        self.o = None   
        self.u = None   
        self.s = None  

        self.u_d = None
        self.s_d = None  
        self.o_d = None 
    
    def fetch(self):
        self.o_value = self.o.get()

fromObject = output()
fromObject2 = output()
toObject = input()

def mixture_of_distribution(a,b):
    return np.dot(a.o,b.o)

toObject.o = fetch_function(mixture_of_distribution, [fromObject,fromObject2])
# toO.o = fromO.o[:,0]+fromO2.o[:,1]

network = [fromObject,fromObject2,toObject]
# <=> input.o = (function of output.o)

fromObject.o = np.array([
    [0.5,0.2],
    [0.2,0.1]
])
fromObject2.o = np.array([
    [1.0,0.0],
    [0.0,0.0]
])
toObject.fetch()

print(toObject.o_value)

# # print(toObject.o_value)

import copy

network2 = copy.deepcopy(network)
network2[1].o = np.array([
    [1.0,1.0],
    [1.0,1.0]
])
network2[2].fetch()
print("Copied network o value :")
print(network2[2].o_value)
print("Original network o value :")
print(network[2].o_value)

toObject.fetch()

print(toObject.o_value)
