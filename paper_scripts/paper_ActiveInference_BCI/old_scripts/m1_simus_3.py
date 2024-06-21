import numpy as np
import statistics as stat
import scipy.stats as scistats
import math,sys,os
import pickle 
import matplotlib.pyplot as plt

import actynf
from tools import clever_running_mean,color_spectrum
from paper_scripts.paper_ActiveInference_BCI.tools_trial_plots import trial_plot_figure

from paper_scripts.paper_ActiveInference_BCI.m1_model import neurofeedback_training

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

def extract_training_data(savepath):
    # EXTRACT TRAINING CURVES    
    with open(savepath, 'rb') as handle:
        saved_data = pickle.load(handle)
    stms = saved_data["stms"]
    weights = saved_data["matrices"]

    Nsubj = len(stms)
    Ntrials = len(weights[0])-1 # One off because we save the initial weights (= trial 0)
    return stms,weights,Nsubj,Ntrials

if __name__ == "__main__":
    Nsubj = 5
    Ntrials = 300
    action_selection_inverse_temp = 32.0
    belief_feedback_std = [0.01]
    true_feedback_std = [0.8,1.0,2.0]#2.0,5.0

    for belief_fb_std in belief_feedback_std:
        for true_fb_std in true_feedback_std:
            learn_a = False 
            clamp_gaussian = False
            
            T = 10
            Th = 2
            feedback_resolution = 5

            subj_cognitive_resolution = 5
            true_cognitive_resolution = 5

            k1b = 0.01
            epsilon_b = 0.01

            k1a = 10
            epsilon_a = 1.0/101.0 # a0 = norm((1/101)* ones + gaussian_prior)*k1a

            k1d = 1.0
            epsilon_d = 1.0

            neutral_action_prop = 0.2 # 20% of the actions have no interest for the task

            pLow = 0.5   # Without any increasing action, there is a pLow chance that the cognitive state will decrease spontaneously
            pUp  = 0.99

            Nsubj = 10


            net = neurofeedback_training(T,Th,  # Trial duration + temporal horizon
                    subj_cognitive_resolution,true_cognitive_resolution,       # Subject belief about cognitive resolution / true cognitive resolution
                    feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
                    belief_feedback_std,true_fb_std,   # Subject belief about feedback noise / true feedback noise
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

            savepath = os.path.join("simulation_outputs","paper1","stairs","long_term","no_learn_a",str(action_selection_inverse_temp),"subject_expects_feedback_std_"+str(belief_fb_std)+"_noClamp","simulations_3."+str(true_fb_std)+".pickle")

            simulate_and_save(net,savepath,Nsubj,Ntrials,override=False)

    
    for alearning in ["no_learn_a"]:

        fig3 = plt.figure(constrained_layout=True)

        widths = [1, 1, 3]
        k = int(len(true_feedback_std)/2.0)
        heights = [1 for i in range(len(true_feedback_std))]
        spec = fig3.add_gridspec(ncols=3, nrows=len(true_feedback_std), 
                            width_ratios=widths,
                            height_ratios=heights)
        o_ax = fig3.add_subplot(spec[:k,2])
        if (len(true_feedback_std)%2==0):
            s_ax = fig3.add_subplot(spec[k:,2])
        else :
            s_ax = fig3.add_subplot(spec[(k+1):,2])

        colorlist = [color_spectrum(np.array([1.0,0.0,0.0]),np.array([0.0,0.0,1.0]),t) for t in np.linspace(0,1,len(true_feedback_std))]
        for id2,belief_fb_std in enumerate(belief_feedback_std):
            for id1,true_fb_std in enumerate(true_feedback_std):
                savepath = os.path.join("simulation_outputs","paper1","stairs",alearning,str(action_selection_inverse_temp),"subject_expects_feedback_std_"+str(belief_fb_std)+"_noClamp","simulations_3."+str(true_fb_std)+".pickle")
                savepath = os.path.join("simulation_outputs","paper1","stairs","long_term",alearning,str(action_selection_inverse_temp),"subject_expects_feedback_std_"+str(belief_fb_std)+"_noClamp","simulations_3."+str(true_fb_std)+".pickle")

                _stm,_weight,_Nsubj,_Ntrials = extract_training_data(savepath)
                color = colorlist[id1]
                full_color = np.concatenate([color,np.array([1.0])],axis=0)
                trans_color = np.concatenate([color,np.array([0.2])],axis=0)

                matrix_true_ax = fig3.add_subplot(spec[id1,0])
                matrix_belief_ax = fig3.add_subplot(spec[id1,1])

                # Extract parameters used
                subj = 0
                trial = 0
                a_img_process = _weight[subj][trial][0]['a'][0]    
                a_img_model = _weight[subj][trial][1]['a'][0]

                # Extract training curves
                arr_o = np.zeros((Nsubj,Ntrials))
                arr_s = np.zeros((Nsubj,Ntrials))
                for subj in range(Nsubj):
                    for trial in range(1,Ntrials):
                        arr_o[subj,trial] = np.mean(_stm[subj][trial][0].o[0])
                        arr_s[subj,trial] = np.mean(_stm[subj][trial][0].x[0])
                arr_o = arr_o[:,1:]
                arr_s = arr_s[:,1:]
                Xs = np.linspace(1,Ntrials,Ntrials-1)

                

                meanarr = np.mean(arr_o,0)
                stdarr = np.std(arr_o,axis=0)
                print(meanarr)
                o_ax.plot(Xs,meanarr,color=full_color,linewidth=2.0)
                o_ax.fill_between(Xs,meanarr-stdarr,meanarr+stdarr,color = trans_color)  

                meanarr = np.mean(arr_s,0)
                stdarr = np.std(arr_s,axis=0)
                s_ax.plot(Xs,meanarr,color=full_color,linewidth=2.0)
                s_ax.fill_between(Xs,meanarr-stdarr,meanarr+stdarr,color = trans_color) 


                matrix_true_ax.imshow(actynf.normalize(a_img_process),vmin=0,vmax=1)
                matrix_belief_ax.imshow(actynf.normalize(a_img_model),vmin=0,vmax=1)
                
        o_ax.set_ylim([0.0,4.0])
        s_ax.set_ylim([0.0,4.0])
        o_ax.grid()
        s_ax.grid()
    plt.show()
    