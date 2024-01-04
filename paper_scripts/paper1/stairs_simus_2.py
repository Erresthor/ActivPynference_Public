import numpy as np
import statistics as stat
import scipy.stats as scistats
import math,sys,os
import pickle 
import matplotlib.pyplot as plt

import actynf
from tools import clever_running_mean,color_spectrum
from stairs_model import neurofeedback_training

# Useful function, might add it to the network class !
def simulate_and_save(my_net,savepath,Nsubj,Ntrials,override=False):
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))

    exists = os.path.isfile(savepath)
    if (not(exists)) or (override):
        stm_subjs = []
        weight_subjs = []
        print("Saving to " + savepath)
        for sub in range(Nsubj):
            subj_net = my_net.copy_network(sub)

            STMs,weights = subj_net.run_N_trials(Ntrials,return_STMs=True,return_weights=True)
            stm_subjs.append(STMs)
            weight_subjs.append(weights)

        save_this = {
            "stms": stm_subjs,
            "matrices" : weight_subjs
        }
            
        with open(savepath, 'wb') as handle:
            pickle.dump(save_this, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved to :   " + savepath)


if __name__ == "__main__":
    Nsubj = 20
    Ntrials = 50
    belief_feedback_std = 0.01
    true_feedback_std = 0.01


    learn_a = False 

    
    clamp_gaussian = False
    action_selection_inverse_temp = 32.0

    T = 10
    Th = 2
    feedback_resolution = 5

    subj_cognitive_resolution = 5
    true_cognitive_resolution = 5

    pre_learnt_action_belief = 0.0
    initial_action_conf = 0.01

    # a0 = norm(ones + a_prior_predominance*gaussian_prior)*initial_feedback_confidence
    a_prior_predominance = 100.0 
    initial_feedback_confidence = 10 

    d_prior_predominance = 1.0 # d0 = norm(ones + d_prior_predominance*first_state)*initial_d_confidence
    initial_d_confidence = 1.0 # here, d_subj = [0.33,0.16,0.16,0.16,0.16] <=> a weak a priori that our starting state is rather low

    neutral_action_prop = 0.2 # 20% of the actions have no interest for the task

    pLow = 0.5   # Without any increasing action, there is a pLow chance that the cognitive state will decrease spontaneously
    pUp  = 0.99

   
    net = neurofeedback_training(T,Th,  # Trial duration + temporal horizon
                    subj_cognitive_resolution,true_cognitive_resolution,       # Subject belief about cognitive resolution / true cognitive resolution
                    feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
                    belief_feedback_std,true_feedback_std,   # Subject belief about feedback noise / true feedback noise
                    neutral_action_prop,       # how many actions have no impact on the state ?
                    [pre_learnt_action_belief,initial_action_conf],  # Action mapping previous knowledge
                    [a_prior_predominance,initial_feedback_confidence],   # Feedback mapping previous knowledge
                    [d_prior_predominance,initial_d_confidence],   # d mapping previous knowledge
                    pLow,pUp,   # How likely it is that the cognitive state will go down when unattended
                                # / how likely it is that the correct action will increase the cognitive state
                    clamp_gaussian=clamp_gaussian,asit = action_selection_inverse_temp) 
                                # Weither to increase the categorical probabilistic weights
                                # on the edges or not
    # savepath = os.path.join(basepath,"simulations_3."+str(true_feedback_std)+".pickle")

    savepath = os.path.join("simulation_outputs","paper1","stairs","no_learn_a",str(action_selection_inverse_temp),"subject_expects_feedback_std_"+str(belief_feedback_std)+"_noClamp","simulations_3."+str(true_feedback_std)+".pickle")

    simulate_and_save(net,savepath,Nsubj,Ntrials,override=False)
