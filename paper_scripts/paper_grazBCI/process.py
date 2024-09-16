# Here, we mer
import numpy as np
import math

import actynf
import actynf.jaxtynf
from tools import gaussian_to_categorical,gaussian_from_distance_matrix,clever_running_mean

import itertools

from functools import partial
import jax
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import tree_map
from jax.scipy.special import gammaln,digamma,betaln
from tensorflow_probability.substrates import jax as tfp
from functools import partial


import jax.numpy as jnp
from jax import grad,jit
from jax.scipy.stats import norm




@partial(jit, static_argnames=["num_bins"])
def discretize_normal_pdf(mean, std, num_bins, lower_bound, upper_bound):
    """ Thank you ChatGPT ! """
    
    # Define the bin edges
    bin_edges = jnp.linspace(lower_bound, upper_bound, num_bins + 1)
    
    # Calculate bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Calculate PDF values at bin centers
    pdf_values = norm.pdf(bin_centers, loc=mean, scale=std)
    
    # Normalize the PDF values to sum to 1 (so it acts like a discrete distribution)
    pdf_values_normalized = pdf_values / jnp.sum(pdf_values)
    
    return pdf_values_normalized, bin_edges

def get_initial_state_priors(Ns,initial_strengths):
    # We have no real prior about the starting state in the series :
    pD = [jnp.ones((ns,))*d_str for (ns,d_str) in zip(Ns,initial_strengths)]
    return pD

def get_emission_priors(Nos,Ns,
                        sensor_noise,
                        prior_strengths,flat_strengths,
                        l_erd_int = True):
    # sensor_noise = [0.5,0.5,0.25,0.25] # Our priors about the noise for each observation modality
    # sensor_prior_confidence = [10.0,10.0,10.0,10.0] # Noise parameters for each observation modality

    # No_bold = [5,5]           
    # No_eeg = [5,5]

    # Emission model : the two hidden states are the orientation and the intensity of the subjects right ERD. 

    def ERDs(normed_orientation,normed_intensity,rest_cst=1.0):
            """ 
            We assume that the ERDs in the left vs right sensorimotor area are drivent by two cognitive components:
            - Intensity 
            - Orientation
            """
            PI = 3.1415
            l_erd = normed_intensity*jnp.cos(normed_orientation*PI/2.0) + rest_cst  # between krest and krest + 1.0
            r_erd = normed_intensity*jnp.sin(normed_orientation*PI/2.0) + rest_cst
            return r_erd,l_erd

    def get_aai(normed_orientation,normed_intensity,krest):
            """
            A cognitive to feedback model function, based only on the laterality of the right hand motor brain activity.
            2 mental states :
            - Intensity
            - Orientation -- normed_orientation is a measure of angle between 0 and 90°   
            We have : 
            rh_mi = int * cos(ori)
            lh_mi = int * sin(ori)  
            """
            r_erd,l_erd = ERDs(normed_orientation,normed_intensity,krest)
            aai =  (l_erd - r_erd)/(l_erd + r_erd)  # Minus
            return aai  # Normalized to -1 -> 1
            # return (aai+1)/2.0  # Normalized to 0 -> 1
            

    def laterality_feedback(_int,_ori,_std,_n_feedback_bins,_base_value=0.01):
        
        # The laterality feedback is a noisy estimator based on the ERD assymetry : 
        _feedback_mean = get_aai(_ori,_int,_base_value)
        
        discretized_feedback,_ = discretize_normal_pdf(_feedback_mean,_std,_n_feedback_bins,-1.0,1.0)

        return discretized_feedback


    def intensity_feedback_l_erd(_int,_ori,_std,_n_feedback_bins,_base_value=0.01):
        # The intensity feedback is a noisy estimator based on the Left ERD intensity : 
        r_erd,l_erd = ERDs(_ori,_int,_base_value)
        
        _feedback_mean = l_erd - _base_value
        
        discretized_feedback,_ = discretize_normal_pdf(_feedback_mean,_std,_n_feedback_bins,0.0,1.0)

        return discretized_feedback
    
    
    def intensity_feedback(_int,_ori,_std,_n_feedback_bins,_base_value=0.01):
        _feedback_mean = _int
        
        discretized_feedback,_ = discretize_normal_pdf(_feedback_mean,_std,_n_feedback_bins,0.0,1.0)

        return discretized_feedback
    

    states_intensity = jnp.linspace(0,1.0,Ns[0])
    states_orientation = jnp.linspace(0,1.0,Ns[1])

    pA = []
    type = ["int","ori","int","ori"]
    sensor = ["eeg","eeg","fmri","fmri"]

    for modality,(n_out_m,prior_noise_m,prior_weight_m,flat_weight_m) in enumerate(zip(Nos,sensor_noise,prior_strengths,flat_strengths)):
        
        
        if type[modality]=="ori":
            emission_builder = lambda x,y: laterality_feedback(x,y,prior_noise_m,n_out_m)
        else :
            if l_erd_int :
                emission_builder = lambda x,y: intensity_feedback_l_erd(x,y,prior_noise_m,n_out_m)
            else:
                emission_builder = lambda x,y: intensity_feedback(x,y,prior_noise_m,n_out_m)
            
        feedback_matrix = vmap(vmap(emission_builder,in_axes=(None,0)),in_axes=(0,None))(states_intensity,states_orientation)
        feedback_matrix_reshaped = jnp.moveaxis(feedback_matrix,-1,0)
        
        feedback_matrix = feedback_matrix_reshaped*prior_weight_m + jnp.ones_like(feedback_matrix_reshaped)*flat_weight_m
        pA.append(feedback_matrix)
    return pA


def _old_build_matrix_with_decay(action,p_low,Ns,rest_state):
    # How to get a decay ? Use the transitions from one state to a rest position :
    starting_states = jnp.arange(Ns)

    # Depending on the action u performed, if there is no decay, 
    # the state will be s+u :
    states_after_action = jnp.clip(starting_states + action,0,Ns-1)

    # There may be a decay after a given action. It is a spontaneous drive towards
    # a "resting" cognitive state. The direction of this decay depends on 
    # the initial state : 
    direction_of_decay_func = jnp.clip(rest_state-starting_states,-1,1)
    
    # States predicted by the action AND the decay
    states_after_action_and_decay = jnp.clip(states_after_action+direction_of_decay_func,0,Ns-1)
    
    # We transform each of these states into matrices using one_hot vectors : 
    action_effect = jnp.swapaxes(jax.nn.one_hot(states_after_action,Ns),0,1)
    decay_effect = jnp.swapaxes(jax.nn.one_hot(states_after_action_and_decay,Ns),0,1)
    
    return action_effect*(1-p_low) + decay_effect*p_low


def build_basic_transition_matrix(action,p_effect,Ns):
    # How to get a decay ? Use the transitions from one state to a rest position :
    starting_states = jnp.arange(Ns)

    # Depending on the action u performed, if there is no decay, 
    # the state will be s+u :
    states_after_action = jnp.clip(starting_states + action,0,Ns-1)
    action_effect = jnp.swapaxes(jax.nn.one_hot(states_after_action,Ns),0,1)

    return action_effect*p_effect + jnp.eye(Ns)*(1-p_effect)


def build_drift_matrix(p_lows,Ns,rest_state):
    # p_lows is a fixed shape array of transition probabilities
    
    # The drift matrix is a set of transitions towards the drift state added together & normalized
    starting_states = jnp.arange(Ns)
    
    abs_potential_transitions = jnp.arange(Ns)
    direction_of_decay_func = jnp.clip(rest_state-starting_states,-1,1)
    
    # All lvl transitions due to drift: (here, it is possible to "drift beyond" the resting state)
    # We may implement an additionnal clipping part somewhere if this is not wanted
    possible_drift_size = p_lows.shape[0]
    levels = jnp.arange(1,possible_drift_size+1)
    
    states_after_lvl_drifts = vmap(lambda lvl : lvl*direction_of_decay_func + starting_states)(levels)
    states_after_lvl_drifts = jnp.clip(states_after_lvl_drifts,0,Ns-1)
    
    vec_states_after_lvl_drifts = jnp.swapaxes(jax.nn.one_hot(states_after_lvl_drifts,Ns),0,-1)
    
    return jnp.einsum("iju,u->ij",vec_states_after_lvl_drifts,p_lows)

def build_full_transition_mapping_with_drift(actions,p_effect,p_lows,Ns,rest_state) :
    # If the subject is idle in a cognitive dimension, there is a 
    # drift towards resting states : The direction of this "decay" depends on 
    # the initial state : 
    
    # Get the transition matrix for all actions : 
    all_transitions = vmap(build_basic_transition_matrix,in_axes=(0,None,None),out_axes=-1)(actions,p_effect,Ns)
    
    # Get the drift matrix for idle actions :
    drift_mat = build_drift_matrix(p_lows,Ns,rest_state)

    
    # Get some mask that will tell us this action is a neutral action :
    filter_idle = jnp.clip(1.0 - jnp.abs(actions),0)
    
    # If this is an idle action, the drift occurs and 
    # the density of the "intended" transition is reduced : 
    density_drift = jnp.sum(p_lows)*filter_idle
    density_intended = 1.0 - density_drift
    
    intended_part = jnp.einsum("iju,u->iju",all_transitions,density_intended) 
    drift_part = jnp.einsum("ij,u->iju",drift_mat,filter_idle)
    return intended_part + drift_part
    
    

def get_transition_priors(Ns,p_drift,p_effect,action_ranges,resting_states,
                          stickinesses,flat_strengths):
    
    # # action_ranges = [jnp.array([-2,-1,0,1,2]),jnp.array([-2,-1,0,1,2])]
    # action_ranges = [jnp.array([-2,-1,0,1,2]),jnp.array([-2,-1,0,1,2])]
    # # action_ranges = [jnp.array([-1,0,1]),jnp.array([-1,0,1])]
    # p_drift = [jnp.array([0.5,0.2]),jnp.array([0.5,0.2])]
    # p_effect = [1.00,1.00]
    # Ns = [5,5]
    # resting_states = [0,2]
    
    pB = []
    for act,pd,pe,ns,rs,sticki,flati in zip(action_ranges,p_drift,p_effect,Ns,resting_states,stickinesses,flat_strengths):
        M =  build_full_transition_mapping_with_drift(act,pe,pd,ns,rs)
        pB.append(sticki*M + flati*jnp.ones_like(M) )
    return pB

if __name__=="__main__":
    pB = get_transition_priors([4,5],[0.5,0.3],[jnp.array([-1,0,1]),jnp.array([-2,0,2])],[10.0,10.0],[1.0,1.0])
    print(pB)
    print([b.shape for b in pB])