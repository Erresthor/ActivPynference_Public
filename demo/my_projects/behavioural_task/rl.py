import numpy as np, matplotlib.pyplot as plt, sys,os

from actynf.base.function_toolbox import normalize
from actynf.layer.model_layer import mdp_layer
from actynf.layer.layer_link import establish_layerLink
from actynf.architecture.network import network

number_of_ticks = 5

def angle_hypothesis():
    """
    9 possible actions (angle between points): not an angle, 0, 45, 90, 135, 180, 225, 270, and 315
    """ 

def model(number_of_ticks,action_model="angle",
          initial_action_mapping_confidence = 0.1):
    '''
    A function defining a generic neurofeedback model depending on a few criteria:
    cognitive_layout : a list of the cognitive dimensions we wish to model
    feedback_Nticks : how we wish to discretize the feedback
    '''

    d0 = normalize(np.ones((number_of_ticks,))) # Uninformed prior on feedback before the task starts
    d = [d0]

    # The feedback is a one-dimensionnal information related to the cognitive dimensions
    feedback = np.zeros((number_of_ticks,Ns))
    for k in range(number_of_ticks):
        feedback[k,k] = 1.0 # Assume that the feedback seen is directly related to the hidden state
                            # (no hidden state)
        # Default : dimension 0 !
    a = [feedback]

    # Action modalities : there are various ways of modeling those actions. 
    # Here, we consider that the subject may entertain 3 different models : 
    # 1. A model where the angle drives the feedback  (9 actions)
    # 2. A model where the position of the point drives the feedback (9 actions)
    # 3. A model where the distance between points drives the feedback (3 actions)
    Ns = number_of_ticks
    if (action_model=="angle"):
        b0 = np.zeros((Ns,Ns,9))
        # If true mapping :
        b0[:,:,0] = np.eye(Ns)  # Angle 0 = no angle
        b0[:,:,1] = np.eye(Ns)  # Angle 1 = around 0°
    elif(action_model=="position"):
        b0 = np.zeros((Ns,Ns,9))
    elif(action_model=="distance"):
        b0 = np.zeros((Ns,Ns,3))

    # If naive model :
    b0 = np.ones(b0.shape)*initial_action_mapping_confidence
     


    # Linear preference matrix c = ln p(o)
    c = [np.linspace(0,number_of_ticks-1,number_of_ticks)]

    e = None
    u = None
    return a,b,c,d,e,u