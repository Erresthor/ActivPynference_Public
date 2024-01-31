import numpy as np
import statistics as stat
import scipy.stats as scistats
import math,sys,os,inspect
import pickle 
import matplotlib.pyplot as plt

import actynf

from tools import clever_running_mean,color_spectrum
from m1_model import neurofeedback_training

#!/usr/bin/python
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# Generate a succession of trial results for a model in a list
# Made to work as a command and simulate a single model index
# To integrate in a cluster like environment

def generate_a_parameter_list(A_priors,a_priors) :
    # Undordered dictionnaries are soooo not cool :(
    simulation_index_list = []    
    for kA in range(len(A_priors)):
        for ka in range(len(a_priors)):
            this_sim = {
                "filename" : "_" + str(A_priors[kA]) + "_" + str(a_priors[ka])+"_.simu",
                "A_std" : A_priors[kA],
                "a_std" : a_priors[ka]
            }

            simulation_index_list.append(this_sim)
    return simulation_index_list
    
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
    input_arguments = sys.argv
    assert len(input_arguments)>=3,"Data generator needs at least 3 arguments : savepath and an index + overwrite"
    name_of_script = input_arguments[0]
    save_folder = input_arguments[1]
    list_index = int(input_arguments[2])
    try :
        overwrite = (input_arguments[3]== 'True')
    except :
        overwrite = False
    
    # PARRALLELIZED SIMULATION OPTIONS
    K = 20
    true_fb_std_array = list(np.linspace(0.001,1.2,K))
    belief_fb_std_array = list(np.linspace(0.001,1.2,K))
    full_parameter_list = generate_a_parameter_list(true_fb_std_array,belief_fb_std_array)
    try :
        param_dict = full_parameter_list[list_index]
        a_std = param_dict["a_std"]
        A_std = param_dict["A_std"]
        filename = param_dict["filename"]
        savepath = os.path.join(save_folder,filename)

    except : # Exit if sim index is beyond length
        print("[" + name_of_script + "] - Index beyond model list length")
        sys.exit()
    
    # FIXED SIMULATION OPTIONS
    Nsubjects = 10
    Ntrials = 100

    learn_a = True 
    clamp_gaussian = False
            
    T = 10
    Th = 2
    feedback_resolution = 5

    subj_cognitive_resolution = 5
    true_cognitive_resolution = 5

    k1b = 0.01
    epsilon_b = 0.01

    k1a = 10.0
    epsilon_a = 1.0/101.0 # a0 = norm((1/101)* ones + gaussian_prior)*k1a

    k1d = 1.0
    epsilon_d = 1.0

    neutral_action_prop = 0.2 # 20% of the actions have no interest for the task

    pLow = 0.5   # Without any increasing action, there is a pLow chance that the cognitive state will decrease spontaneously
    pUp  = 0.99

    action_selection_inverse_temp = 32.0


    net = neurofeedback_training(T,Th,  # Trial duration + temporal horizon
            subj_cognitive_resolution,true_cognitive_resolution,       # Subject belief about cognitive resolution / true cognitive resolution
            feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
            a_std,A_std,   # Subject belief about feedback noise / true feedback noise
            neutral_action_prop,       # how many actions have no impact on the state ?
            k1b,epsilon_b,  # Action mapping previous knowledge
            k1a,epsilon_a,   # Feedback mapping previous knowledge
            k1d,epsilon_d,   # d mapping previous knowledge
            pLow,pUp,   # How likely it is that the cognitive state will go down when unattended
                        # / how likely it is that the correct action will increase the cognitive state
            clamp_gaussian=clamp_gaussian,asit = action_selection_inverse_temp,
            learn_a=learn_a) 
                        # Clamp : Weither to increase the categorical probabilistic weights
                        # on the edges or not
                        # asit : inverse temperature of the action selection process
                        # learn_a : Weither to learn the perception matrix on the go                                       

    simulate_and_save(net,savepath,Nsubjects,Ntrials,override=overwrite)
    print("[" + name_of_script + "] - Saving model results at :   " + savepath)