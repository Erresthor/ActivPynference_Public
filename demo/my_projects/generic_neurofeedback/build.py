import numpy as np, matplotlib.pyplot as plt, sys,os

from actynf.base.function_toolbox import normalize
from actynf.layer.model_layer import mdp_layer
from actynf.layer.layer_link import establish_layerLink
from actynf.architecture.network import network

def generic_neurofeedback_weights(cognitive_layout,feedback_Nticks,
                                  feedback_based_on_dimensions=None):
    '''
    A function defining a generic neurofeedback model depending on a few criteria:
    cognitive_layout : a list of the cognitive dimensions we wish to model
    feedback_Nticks : how we wish to discretize the feedback
    '''

    d = []
    for cognitive_index,cognitive_dimension in enumerate(cognitive_layout):
        # For now, let's assume we may start in any combination of states
        d.append(normalize(np.ones(cognitive_dimension,)))
    
    # Let's assume there exist all the needed actions to get to any state of any factor
    b = []
    for cog_dim_idx,cog_dim_size in enumerate(cognitive_layout):
        b_factor = np.zeros((cog_dim_size,cog_dim_size,cog_dim_size)) # From any cognitive state, we can go to any other
        for from_factor in range(cog_dim_size):
            for action_idx in range(cog_dim_size):
                b_factor[action_idx,from_factor,action_idx] = 1.0
        b.append(b_factor)
    # Warning ! If the number of states / factors is too big, the agent will have trouble
    # to explore correctly !
    

    # The feedback is a one-dimensionnal information related to the cognitive dimensions
    feedback = np.zeros((feedback_Nticks,)+tuple(cognitive_layout))
    if (feedback_based_on_dimensions==None):
        # Default : dimension 0 !
        
    # Depending on the feedback options


    a = [feedback]

    # Linear preference matrix c = ln p(o)
    c = [np.linspace(0,feedback_Nticks-1,feedback_Nticks)]

    e = None
    u = None
    return a,b,c,d,e,u