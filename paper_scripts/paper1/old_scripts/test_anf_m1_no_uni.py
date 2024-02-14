# Make basic package imports
import numpy as np
import statistics as stat
import scipy.stats as scistats
import math,sys,os,inspect
import pickle 
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

# Import actynf and the functions to help us plot the results
import actynf
print(actynf.__version__)
from tools import simulate_and_save,extract_training_data # Saving and loading simulation outputs
from tools import dist_kl_dir # A tool to qualify subject learning
from tools import clever_running_mean,color_spectrum # Plotting helpers
from tools_trial_plots import plot_training_curve # Plotting helpers
from test_m1_model import neurofeedback_training # The Active Inference model we use



SAVING_FOLDER = os.path.join("..","..","simulation_outputs","paper1","TESTS")
print("Results will be saved to " + SAVING_FOLDER + " (relative to this file)")

# Functions defining the model and plotting the results !

def run_and_save_model(savepath,
        true_feedback_std,belief_feedback_std,
        Nsubj,Ntrials,
        overwrite=False):
    """
    Run a simulation of model m1 using the following parameters : 
    - savepath : where to store simulation outputs
    - true_feedback_std : wanted sigma_process value
    - belief_feedback_std : wanted sigma_model value for the feedback
    - Nsubj : subjects simulated for the simulations
    - Ntrials : how many trials simulated per subject (duration of the training)
    - overwrite : run simulations even if there exist results already ?
    """
    learn_a = False # In this first simulation, the subjects won't be questionning their feedback mappings

    T = 10
    Th = 2
    feedback_resolution = 5 # No

    subj_cognitive_resolution = 5 # Ns (model)
    true_cognitive_resolution = 5 # Ns (process)

    k1b = 0.01   # Initial action mapping weight
    epsilon_b = 0.01

    k1a = 10     # Initial feedback mapping weight (model)
    epsilon_a = 1.0/101.0   # Added uniform distribution weight added
        # a0 is defined as = norm(epsilon_a* ones + gaussian_prior)*k1a

    k1d = 1.0   # Initial starting state mapping weight (model)
    epsilon_d = 1.0  # Added uniform distribution weight added
        # d0 is defined as = norm(epsilon_d* ones + [1,1,0,0,0])*k1d   [for Ns=5]
    
    neutral_action_prop = 0.2 # 20% of the actions have no effect on the subject cognitive state

    pRest = 0.5   # Without any increasing action, there is a pRest chance that the cognitive state will decrease spontaneously
    pEffect  = 0.99   # An adequate action will increase the subject mental state with a probability pEffect

    action_selection_inverse_temp = 32.0 # How much noise in the selection of actions after EFE calculation
    
    net = neurofeedback_training(T,Th,  # Trial duration + temporal horizon
            subj_cognitive_resolution,true_cognitive_resolution,       # Subject belief about cognitive resolution / true cognitive resolution
            feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
            belief_feedback_std,true_feedback_std,   # Subject belief about feedback noise / true feedback noise
            neutral_action_prop,       # how many actions have no impact on the state ?
            k1b,epsilon_b,  # Action mapping previous knowledge
            k1a,epsilon_a,   # Feedback mapping previous knowledge
            k1d,epsilon_d,   # d mapping previous knowledge
            pRest,pEffect,   # How likely it is that the cognitive state will go down when unattended
                        # / how likely it is that the correct action will increase the cognitive state
            clamp_gaussian=False,asit = action_selection_inverse_temp,
            learn_a=learn_a) 
                        # Clamp : Weither to increase the categorical probabilistic weights
                        # on the edges or not
                        # asit : inverse temperature of the action selection process
                        # learn_a : Weither to learn the perception matrix on the go                                       

    # print(net.layers[1].a[0])
    # plt.imshow(actynf.normalize(net.layers[1].a[0]),vmin=0.0,vmax=1.0)
    # plt.show()
    
    simulate_and_save(net,savepath,Nsubj,Ntrials,override=overwrite)

def run_and_save_model_absolute(savepath,
        true_feedback_std,belief_feedback_std,
        Nsubj,Ntrials,
        epsilon_a=0.0,
        overwrite=False):
    """
    Run a simulation of model m1 using the following parameters : 
    - savepath : where to store simulation outputs
    - true_feedback_std : wanted sigma_process value
    - belief_feedback_std : wanted sigma_model value for the feedback
    - Nsubj : subjects simulated for the simulations
    - Ntrials : how many trials simulated per subject (duration of the training)
    - overwrite : run simulations even if there exist results already ?
    """
    learn_a = True # In this first simulation, the subjects won't be questionning their feedback mappings

    T = 10
    Th = 2
    feedback_resolution = 5 # No

    subj_cognitive_resolution = 5 # Ns (model)
    true_cognitive_resolution = 5 # Ns (process)

    k1b = 0.01   # Initial action mapping weight
    epsilon_b = 0.01

    k1a = 10     # Initial feedback mapping weight (model)
    # epsilon_a = 0.0/101.0   # Added uniform distribution weight added
        # a0 is defined as = norm(epsilon_a* ones + gaussian_prior)*k1a

    k1d = 1.0   # Initial starting state mapping weight (model)
    epsilon_d = 1.0  # Added uniform distribution weight added
        # d0 is defined as = norm(epsilon_d* ones + [1,1,0,0,0])*k1d   [for Ns=5]
    
    neutral_action_prop = 0.2 # 20% of the actions have no effect on the subject cognitive state

    pRest = 0.5   # Without any increasing action, there is a pRest chance that the cognitive state will decrease spontaneously
    pEffect  = 0.99   # An adequate action will increase the subject mental state with a probability pEffect

    action_selection_inverse_temp = 32.0 # How much noise in the selection of actions after EFE calculation
    
    net = neurofeedback_training(T,Th,  # Trial duration + temporal horizon
            subj_cognitive_resolution,true_cognitive_resolution,       # Subject belief about cognitive resolution / true cognitive resolution
            feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
            belief_feedback_std,true_feedback_std,   # Subject belief about feedback noise / true feedback noise
            neutral_action_prop,       # how many actions have no impact on the state ?
            k1b,epsilon_b,  # Action mapping previous knowledge
            k1a,epsilon_a,   # Feedback mapping previous knowledge
            k1d,epsilon_d,   # d mapping previous knowledge
            pRest,pEffect,   # How likely it is that the cognitive state will go down when unattended
                        # / how likely it is that the correct action will increase the cognitive state
            clamp_gaussian=False,asit = action_selection_inverse_temp,
            learn_a=learn_a) 
                        # Clamp : Weither to increase the categorical probabilistic weights
                        # on the edges or not
                        # asit : inverse temperature of the action selection process
                        # learn_a : Weither to learn the perception matrix on the go 
    # net.layers[1].learn_options.backwards_pass = False                       
    # print(net.layers[1].a[0])
    # plt.imshow(actynf.normalize(net.layers[1].a[0]),vmin=0.0,vmax=1.0)
    # plt.show()
    simulate_and_save(net,savepath,Nsubj,Ntrials,override=overwrite)
    

sigma_process = 0.01
sigma_model = 0.5

N_subj = 1
N_trials = 1000

savepath = os.path.join(SAVING_FOLDER,"simulation_test_efe")
run_and_save_model_absolute(savepath,sigma_process,sigma_model,N_subj,N_trials,0.01,True)




_stm,_weight,_Nsubj,_Ntrials = extract_training_data(savepath)
print(_weight[0][-1][1]["a"][0])

action_select_hist = _stm[0][-1][1].Gd

fig2,axs2= plt.subplots(1,10-1)
axs2[0].set_ylabel("Actions")
for t in range(10-1):
    im = axs2[t].imshow(action_select_hist[...,t],vmax = np.max(action_select_hist[...,t]),vmin=np.min(action_select_hist[...,t][action_select_hist[...,t]!=-1000]))
    axs2[t].set_xticks(range(action_select_hist.shape[0]))
    axs2[t].set_xticklabels(["Habits","Exploit","Uncertainty","FB novelty","ACT novelty","Deeper"],rotation=45,fontsize=4)
    fig2.colorbar(im,fraction=0.046, pad=0.04)
fig2.show()

input()


fig = plot_training_curve(_stm,_weight,"")
fig.show()
input()
# subject = 0 # What subject are we studying ?
# plot_these_trials = range(1,40,1)
# for trial in (plot_these_trials):
#     print(np.round(actynf.normalize(_weight[subject][trial][1]["a"][0]),2))
#     plot_one_trial(_stm,0,trial,"Trial " + str(trial))
