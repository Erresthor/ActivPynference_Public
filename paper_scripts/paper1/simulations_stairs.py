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

# Extract saved data
def extract_training_curves(savepath):
    # EXTRACT TRAINING CURVES    
    with open(savepath, 'rb') as handle:
        saved_data = pickle.load(handle)
    stms = saved_data["stms"]
    weights = saved_data["matrices"]

    Nsubj = len(stms)
    Ntrials = len(weights[0])-1 # One off because we save the initial weights (= trial 0)

    observations = []
    mean_obs = []
    for subj in range(Nsubj):
        # print("________________________________________________________________________________________________________________________________")
        observations.append([])
        mean_obs.append([])
        feedback_learn_imgs = []
        for trial in range(1,Ntrials+1):
            o_s = (stms[subj][trial][0].o)
            u_s = (stms[subj][trial][1].u)
            # for t in range(T-1):
            #     if (o_s[0,t] == 3) :
            #         # print("HERE !")
            #         print(u_s[t])

            observations[-1].append(stms[subj][trial][1].o)
            mean_obs[-1].append(np.mean(stms[subj][trial][1].o))
    return observations,mean_obs

def extract_training_data(savepath):
    # EXTRACT TRAINING CURVES    
    with open(savepath, 'rb') as handle:
        saved_data = pickle.load(handle)
    stms = saved_data["stms"]
    weights = saved_data["matrices"]

    Nsubj = len(stms)
    Ntrials = len(weights[0])-1 # One off because we save the initial weights (= trial 0)
    return stms,weights,Nsubj,Ntrials

def plot_smoothed_training_curves(mean_val_list,ax,label_curve,
                                  color = np.array([0.0,1.0,0.0]),
                                  smooth_other = 5,normalization_cst = 4.0):
    full_color = np.concatenate([color,np.array([1.0])],axis=0)
    trans_color = np.concatenate([color,np.array([0.2])],axis=0)
    
    smoothed_mean_obs = [clever_running_mean(subj_obs,smooth_other)/normalization_cst for subj_obs in mean_val_list]
    smoothed_mean_arr = np.array(smoothed_mean_obs)

    Ntrials = smoothed_mean_arr.shape[1]
    Xs = np.linspace(0,Ntrials,Ntrials)
    for ksubj in range(len(mean_val_list)):
        ax.plot(Xs,smoothed_mean_arr[ksubj,:],color=full_color,linewidth=0.2)
    mean_values = np.mean(smoothed_mean_arr,axis=0) # mean of all subjects
    std_vals = np.std(smoothed_mean_arr,axis=0) # std of all subjects
    ax.plot(Xs,mean_values,color=full_color,linewidth=2.0,label=label_curve)
    ax.fill_between(Xs,mean_values-std_vals,mean_values+std_vals,color = trans_color)

# The first plots we did
def subject_expects_low_noise(): # std = 0.1
    basepath = os.path.join("simulation_outputs","paper1","subject_expects_low_noise")
        
    Nsubj = 10
    Ntrials = 100
    T = 10
    Th =2 
    smooth_window = 3
    feedback_resolution = 5

    low_noise_net = neurofeedback_training(T,2,  # Trial duration + temporal horizon
                    5,5,       # Subject belief about cognitive resolution / true cognitive resolution
                    feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
                    0.1,0.1,   # Subject belief about feedback noise / true feedback noise
                    0.2,       # how many actions have no impact on the state ?
                    [0.1,0.01],  # Action mapping previous knowledge
                    [5.0,500],   # Feedback mapping previous knowledge
                    [1.0,1.0],   # d mapping previous knowledge
                    0.5,0.99)   # How likely it is that the cognitive state will go down when unattended
                                # / how likely it is that the correct action will increase the cognitive state
    savepath_low = os.path.join(basepath,"simulations_3.low_noise.pickle")

    medium_noise_net = neurofeedback_training(T,2,  # Trial duration + temporal horizon
                    5,5,       # Subject belief about cognitive resolution / true cognitive resolution
                    feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
                    0.1,0.3,   # Subject belief about feedback noise / true feedback noise
                    0.2,       # how many actions have no impact on the state ?
                    [0.1,0.01],  # Action mapping previous knowledge
                    [5.0,500],   # Feedback mapping previous knowledge
                    [1.0,1.0],   # d mapping previous knowledge
                    0.5,0.99)   # How likely it is that the cognitive state will go down when unattended
                                # / how likely it is that the correct action will increase the cognitive state
    savepath_med = os.path.join(basepath,"simulations_3.medium_noise.pickle")

    high_noise_net = neurofeedback_training(T,2,  # Trial duration + temporal horizon
                    5,5,       # Subject belief about cognitive resolution / true cognitive resolution
                    feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
                    0.1,0.5,   # Subject belief about feedback noise / true feedback noise
                    0.2,       # how many actions have no impact on the state ?
                    [0.1,0.01],  # Action mapping previous knowledge
                    [5.0,500],   # Feedback mapping previous knowledge
                    [1.0,1.0],   # d mapping previous knowledge
                    0.5,0.99)   # How likely it is that the cognitive state will go down when unattended
                                # / how likely it is that the correct action will increase the cognitive state
    savepath_high = os.path.join(basepath,"simulations_3.high_noise.pickle")

    savepath_list = [savepath_low,savepath_med,savepath_high]
    colorlist = [np.array([0.0,0.0,1.0]),np.array([0.5,0.0,0.5]),np.array([1.0,0.0,0.0])]
    netlist = [low_noise_net,medium_noise_net,high_noise_net]

    # Simulate data
    for net,path,color in zip(netlist,savepath_list,colorlist):
        simulate_and_save(net,path,Nsubj,Ntrials,override=False)

    # Plot simulated !
    fig,ax = plt.subplots()
    for net,path,color in zip(netlist,savepath_list,colorlist):
        # EXTRACT TRAINING CURVES (manual opening)
        # with open(path, 'rb') as handle:
        #     saved_data = pickle.load(handle)
        # stms = saved_data["stms"]
        # weights = saved_data["matrices"]

        
        label = path.split(".")[-2]
        observations, mean_obs = extract_training_curves(path)
        plot_smoothed_training_curves(mean_obs,ax,label,color,smooth_window,feedback_resolution-1.0)        
        
    ax.set_xlabel("Trials")
    ax.set_ylabel("Average feedback received")
    ax.legend()
    ax.grid()
    ax.set_ylim(0.0,1.0)
    ax.set_title("Training curve for subject with initial feedback confidence variance = 0.1")
    ax.set_title("Training curve for subject where " + basepath.split('\\')[-1])
    fig.show()

def subject_expects_medium_noise(): # std = 0.5
    basepath = os.path.join("simulation_outputs","paper1","subject_expects_medium_noise")
    Nsubj = 10
    Ntrials = 100
    T = 10
    Th =2 
    smooth_window = 3
    feedback_resolution = 5

    low_noise_net = neurofeedback_training(T,2,  # Trial duration + temporal horizon
                    5,5,       # Subject belief about cognitive resolution / true cognitive resolution
                    feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
                    0.5,0.1,   # Subject belief about feedback noise / true feedback noise
                    0.2,       # how many actions have no impact on the state ?
                    [0.1,0.01],  # Action mapping previous knowledge
                    [5.0,500],   # Feedback mapping previous knowledge
                    [1.0,1.0],   # d mapping previous knowledge
                    0.5,0.99)   # How likely it is that the cognitive state will go down when unattended
                                # / how likely it is that the correct action will increase the cognitive state
    savepath_low = os.path.join(basepath,"simulations_3.low_noise.pickle")

    medium_noise_net = neurofeedback_training(T,2,  # Trial duration + temporal horizon
                    5,5,       # Subject belief about cognitive resolution / true cognitive resolution
                    feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
                    0.5,0.3,   # Subject belief about feedback noise / true feedback noise
                    0.2,       # how many actions have no impact on the state ?
                    [0.1,0.01],  # Action mapping previous knowledge
                    [5.0,500],   # Feedback mapping previous knowledge
                    [1.0,1.0],   # d mapping previous knowledge
                    0.5,0.99)   # How likely it is that the cognitive state will go down when unattended
                                # / how likely it is that the correct action will increase the cognitive state
    savepath_med = os.path.join(basepath,"simulations_3.medium_noise.pickle")

    high_noise_net = neurofeedback_training(T,2,  # Trial duration + temporal horizon
                    5,5,       # Subject belief about cognitive resolution / true cognitive resolution
                    feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
                    0.5,0.5,   # Subject belief about feedback noise / true feedback noise
                    0.2,       # how many actions have no impact on the state ?
                    [0.1,0.01],  # Action mapping previous knowledge
                    [5.0,500],   # Feedback mapping previous knowledge
                    [1.0,1.0],   # d mapping previous knowledge
                    0.5,0.99)   # How likely it is that the cognitive state will go down when unattended
                                # / how likely it is that the correct action will increase the cognitive state
    savepath_high = os.path.join(basepath,"simulations_3.high_noise.pickle")

    savepath_list = [savepath_low,savepath_med,savepath_high]
    colorlist = [np.array([0.0,0.0,1.0]),np.array([0.5,0.0,0.5]),np.array([1.0,0.0,0.0])]
    netlist = [low_noise_net,medium_noise_net,high_noise_net]

    # Simulate data
    for net,path,color in zip(netlist,savepath_list,colorlist):
        simulate_and_save(net,path,Nsubj,Ntrials,override=False)

    # Plot simulated !
    fig,ax = plt.subplots()
    for net,path,color in zip(netlist,savepath_list,colorlist):
        # EXTRACT TRAINING CURVES (manual opening)
        # with open(path, 'rb') as handle:
        #     saved_data = pickle.load(handle)
        # stms = saved_data["stms"]
        # weights = saved_data["matrices"]

        
        label = path.split(".")[-2]
        observations, mean_obs = extract_training_curves(path)
        plot_smoothed_training_curves(mean_obs,ax,label,color,smooth_window,feedback_resolution-1.0)        
        
    ax.set_xlabel("Trials")
    ax.set_ylabel("Average feedback received")
    ax.legend()
    ax.grid()
    ax.set_ylim(0.0,1.0)
    ax.set_title("Training curve for subject where " + basepath.split('\\')[-1])
    fig.show()

def subject_expects_high_noise(): # std = 0.7
    basepath = os.path.join("simulation_outputs","paper1","subject_expects_high_noise")
        
    Nsubj = 10
    Ntrials = 100
    T = 10
    Th =2 
    smooth_window = 3
    feedback_resolution = 5

    low_noise_net = neurofeedback_training(T,2,  # Trial duration + temporal horizon
                    5,5,       # Subject belief about cognitive resolution / true cognitive resolution
                    feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
                    0.7,0.1,   # Subject belief about feedback noise / true feedback noise
                    0.2,       # how many actions have no impact on the state ?
                    [0.1,0.01],  # Action mapping previous knowledge
                    [5.0,500],   # Feedback mapping previous knowledge
                    [1.0,1.0],   # d mapping previous knowledge
                    0.5,0.99)   # How likely it is that the cognitive state will go down when unattended
                                # / how likely it is that the correct action will increase the cognitive state
    savepath_low = os.path.join(basepath,"simulations_3.low_noise.pickle")

    medium_noise_net = neurofeedback_training(T,2,  # Trial duration + temporal horizon
                    5,5,       # Subject belief about cognitive resolution / true cognitive resolution
                    feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
                    0.7,0.3,   # Subject belief about feedback noise / true feedback noise
                    0.2,       # how many actions have no impact on the state ?
                    [0.1,0.01],  # Action mapping previous knowledge
                    [5.0,500],   # Feedback mapping previous knowledge
                    [1.0,1.0],   # d mapping previous knowledge
                    0.5,0.99)   # How likely it is that the cognitive state will go down when unattended
                                # / how likely it is that the correct action will increase the cognitive state
    savepath_med = os.path.join(basepath,"simulations_3.medium_noise.pickle")

    high_noise_net = neurofeedback_training(T,2,  # Trial duration + temporal horizon
                    5,5,       # Subject belief about cognitive resolution / true cognitive resolution
                    feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
                    0.7,0.5,   # Subject belief about feedback noise / true feedback noise
                    0.2,       # how many actions have no impact on the state ?
                    [0.1,0.01],  # Action mapping previous knowledge
                    [5.0,500],   # Feedback mapping previous knowledge
                    [1.0,1.0],   # d mapping previous knowledge
                    0.5,0.99)   # How likely it is that the cognitive state will go down when unattended
                                # / how likely it is that the correct action will increase the cognitive state
    savepath_high = os.path.join(basepath,"simulations_3.high_noise.pickle")

    savepath_list = [savepath_low,savepath_med,savepath_high]
    colorlist = [np.array([0.0,0.0,1.0]),np.array([0.5,0.0,0.5]),np.array([1.0,0.0,0.0])]
    netlist = [low_noise_net,medium_noise_net,high_noise_net]

    # Simulate data
    for net,path,color in zip(netlist,savepath_list,colorlist):
        simulate_and_save(net,path,Nsubj,Ntrials,override=False)

    # Plot simulated !
    fig,ax = plt.subplots()
    for net,path,color in zip(netlist,savepath_list,colorlist):
        # EXTRACT TRAINING CURVES (manual opening)
        # with open(path, 'rb') as handle:
        #     saved_data = pickle.load(handle)
        # stms = saved_data["stms"]
        # weights = saved_data["matrices"]

        
        label = path.split(".")[-2]
        observations, mean_obs = extract_training_curves(path)
        plot_smoothed_training_curves(mean_obs,ax,label,color,smooth_window,feedback_resolution-1.0)        
        
    ax.set_xlabel("Trials")
    ax.set_ylabel("Average feedback received")
    ax.legend()
    ax.grid()
    ax.set_ylim(0.0,1.0)
    ax.set_title("Training curve for subject where " + basepath.split('\\')[-1])
    fig.show()

def var_confidence_variable_true_noise_figure(belief_feedback_std,clamp_gaussian=False,interrupt_early = False):
    basepath = os.path.join("simulation_outputs","paper1","belief_vs_true_fb_std","subject_expects_feedback_std_"+str(belief_feedback_std)+("" if clamp_gaussian else "_noClamp"))
        
    Nsubj = 10
    Ntrials = 100
    T = 10
    Th = 2
    smooth_window = 3
    feedback_resolution = 5

    subj_cognitive_resolution = 5
    true_cognitive_resolution = 5

    pre_learnt_action_belief = 0.0
    initial_action_conf = 0.01

    a_prior_predominance = 100.0 # a0 = norm(ones + 100*gaussian_prior)*500
    initial_feedback_confidence = 10

    label_list = ["true_fb_noise=0.1","true_fb_noise=0.4","true_fb_noise=0.5","true_fb_noise=0.6","true_fb_noise=1.0","true_fb_noise=5.0"]
    colorlist = [np.array([0.0,0.0,1.0]),np.array([0.25,0.0,0.75]),np.array([0.5,0.0,0.5]),np.array([0.75,0.0,0.25]),np.array([1.0,0.0,0.0]),np.array([0.0,0.0,0.0])]
    noise_stds = [0.1,0.4,0.5,0.6,1.0,5.0]
    savepath_list = []
    netlist = []
    for true_fb_noise in noise_stds:
        net = neurofeedback_training(T,Th,  # Trial duration + temporal horizon
                    subj_cognitive_resolution,true_cognitive_resolution,       # Subject belief about cognitive resolution / true cognitive resolution
                    feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
                    belief_feedback_std,true_fb_noise,   # Subject belief about feedback noise / true feedback noise
                    0.2,       # how many actions have no impact on the state ?
                    [pre_learnt_action_belief,initial_action_conf],  # Action mapping previous knowledge
                    [a_prior_predominance,initial_feedback_confidence],   # Feedback mapping previous knowledge
                    [1.0,1.0],   # d mapping previous knowledge
                    0.5,0.99,   # How likely it is that the cognitive state will go down when unattended
                                # / how likely it is that the correct action will increase the cognitive state
                    clamp_gaussian=clamp_gaussian) 
                                # Weither to increase the categorical probabilistic weights
                                # on the edges or not
        savepath = os.path.join(basepath,"simulations_3."+str(true_fb_noise)+".pickle")
        savepath_list.append(savepath)
        netlist.append(net)

    # # Check the shape of the subject prior belief matrices !
    # fig,axs = plt.subplots(2,4)
    # k = 0
    # for net,path,color in zip(netlist,savepath_list,colorlist):
    #     axs[0,k].imshow(actynf.normalize(net.layers[0].a[0]),vmin=0,vmax=1)
    #     axs[0,k].set_title("Feedback noise")
    #     axs[1,k].imshow(actynf.normalize(net.layers[1].a[0]),vmin=0,vmax=1)
    #     axs[1,k].set_title("Subject belief noise")
    #     k = k + 1
    # fig.show()
    # return

    # Simulate data
    for net,path,color in zip(netlist,savepath_list,colorlist):
        simulate_and_save(net,path,Nsubj,Ntrials,override=False)

    if (interrupt_early):
        return
    
    # # Plot simulated data !
    # # Feedback perception matrices for one subject
    # plot_subj_n = 3 
    # for net,path,color,label in zip(netlist,savepath_list,colorlist,label_list):
    #     fig_net,ax3 = plt.subplots(2,2)
    #     label = path.split(".")[-2]
    #     stms,weights,Nsubj,Ntrials = extract_training_data(path)
    #     ax3[0,0].imshow(actynf.normalize(weights[plot_subj_n][0][1]["a"][0]),vmin=0,vmax=1)
    #     ax3[0,0].set_title("Initial")
    #     ax3[1,0].imshow(actynf.normalize(weights[plot_subj_n][-1][1]["a"][0]),vmin=0,vmax=1)
    #     ax3[1,0].set_title("Final")
    #     ax3[0,1].imshow(actynf.normalize(weights[plot_subj_n][0][0]["a"][0]),vmin=0,vmax=1)
    #     ax3[0,1].set_title("True mapping")
    #     fig_net.suptitle("Initial vs final belief about feedback mapping")
    #     fig_net.show()

    # Plot smoothed training curves for all subjects : 
    # True observations :
    fig,ax = plt.subplots()
    for net,path,color,label in zip(netlist,savepath_list,colorlist,label_list):
        full_color = np.concatenate([color,np.array([1.0])],axis=0)
        trans_color = np.concatenate([color,np.array([0.2])],axis=0)
        observations, mean_obs = extract_training_curves(path)
        plot_smoothed_training_curves(mean_obs,ax,label,color,smooth_window,feedback_resolution-1.0)          
    ax.set_xlabel("Trials")
    ax.set_ylabel("Average feedback received")
    ax.legend()
    ax.grid()
    ax.set_ylim(0.0,1.0)
    ax.set_title("Training curve for subject with initial confidence = " + str(belief_feedback_std))
    fig.show()

    # Plot smoothed state training curves for all subjects :
    # true cognitive level
    fig,ax = plt.subplots()
    for net,path,color,label in zip(netlist,savepath_list,colorlist,label_list):
        full_color = np.concatenate([color,np.array([1.0])],axis=0)
        trans_color = np.concatenate([color,np.array([0.2])],axis=0)
        stms,weights,Nsubj,Ntrials = extract_training_data(path)
        all_states = []
        for subj in range(len(stms)):
            all_states.append([])
            for trial in range(1,len(stms[subj]),1):
                all_states[-1].append(np.mean(stms[subj][trial][0].x[0,:]))
        plot_smoothed_training_curves(all_states,ax,label,color,smooth_window,true_cognitive_resolution-1)
    ax.set_xlabel("Trials")
    ax.set_ylabel("Average level of cognition achieved")
    ax.legend()
    ax.grid()
    ax.set_ylim(0.0,1.0)
    ax.set_title("Training curve for subject with initial confidence = " + str(belief_feedback_std))
    fig.show()

    # # Plot averaged inference during a trial
    # Average trial : subject state inference
    fig,ax = plt.subplots(1,2)
    state_scale = np.array(range(subj_cognitive_resolution))
    for net,path,color,label in zip(netlist,savepath_list,colorlist,label_list):
        full_color = np.concatenate([color,np.array([1.0])],axis=0)
        trans_color = np.concatenate([color,np.array([0.2])],axis=0)

        stms,weights,Nsubj,Ntrials = extract_training_data(path)
        state_expectancies = []
        state_entropies = []
        for subj in range(len(stms)):
            state_expectancies.append([])
            state_entropies.append([])

            for trial in range(1,len(stms[subj]),1):
                belief_about_states  = stms[subj][trial][1].x_d[:,:]

                belief_entropy = scistats.entropy(belief_about_states,axis=0)
                belief_expectancy = np.sum(belief_about_states*state_scale[:, np.newaxis],axis=0)
                
                state_expectancies[-1].append(belief_expectancy)
                state_entropies[-1].append(belief_entropy)

        state_expectancy_array = np.array(state_expectancies)/subj_cognitive_resolution
        state_entropy_array = np.array(state_entropies)/subj_cognitive_resolution
        
        T = state_expectancy_array.shape[-1]
        Ts = np.linspace(0,T,T)
        Nsubj = state_expectancy_array.shape[0]
        averaged_first_25_trials = np.mean(state_expectancy_array[:,:25,:],axis=1)
        averaged_first_25_trials_e = np.mean(state_entropy_array[:,:25,:],axis=1)

        averaged_last_25_trials = np.mean(state_expectancy_array[:,-25:,:],axis=1)
        averaged_last_25_trials_e = np.mean(state_entropy_array[:,-25:,:],axis=1)

        for subj in range(Nsubj):
            ax[0].plot(Ts,averaged_first_25_trials[subj,:],color=full_color,linewidth=0.2)
        mean_values = np.mean(averaged_first_25_trials,axis=0) # mean of all subjects
        std_vals = np.std(averaged_first_25_trials,axis=0) # std of all subjects
        ax[0].plot(Ts,mean_values,color=full_color,linewidth=2.0,label=label)
        ax[0].fill_between(Ts,mean_values-std_vals,mean_values+std_vals,color = trans_color)

        for subj in range(Nsubj):
            ax[1].plot(Ts,averaged_last_25_trials[subj,:],color=full_color,linewidth=0.2)
        mean_values = np.mean(averaged_last_25_trials,axis=0) # mean of all subjects
        std_vals = np.std(averaged_last_25_trials,axis=0) # std of all subjects
        ax[1].plot(Ts,mean_values,color=full_color,linewidth=2.0,label=label)
        ax[1].fill_between(Ts,mean_values-std_vals,mean_values+std_vals,color = trans_color)

    ax[0].set_xlabel("Timesteps")
    ax[1].set_xlabel("Timesteps")
    ax[0].set_ylabel("Subject belief expectancy about achieved cognitive state")
    
    ax[0].legend()
    ax[0].grid()
    ax[0].set_ylim(0.0,1.0)
    ax[1].legend()
    ax[1].grid()
    ax[1].set_ylim(0.0,1.0)

    ax[0].set_title("First 25 trials")
    ax[1].set_title("Last 25 trials")
    fig.show()

    # Plot initial weights for feedback (true vs. subject belief)
    plot_subj_ns = [0,1,2]
    for plot_subj_n in plot_subj_ns :
        fig,axs = plt.subplots(3,len(label_list))
        fig.suptitle('Subject ' + str(plot_subj_n))
        k = 0
        for net,path,color,label in zip(netlist,savepath_list,colorlist,label_list):
            stms,weights,Nsubj,Ntrials = extract_training_data(path)
            
            axs[0,k].imshow(actynf.normalize(weights[plot_subj_n][0][0]["a"][0]),vmin=0,vmax=1)
            axs[0,k].set_title(label)
            axs[1,k].imshow(actynf.normalize(weights[plot_subj_n][0][1]["a"][0]),vmin=0,vmax=1)
            axs[1,k].set_title("Initial feedback belief")
            axs[2,k].imshow(actynf.normalize(weights[plot_subj_n][-1][1]["a"][0]),vmin=0,vmax=1)
            axs[2,k].set_title("Final feedback belief")
            k = k + 1
        fig.show()
    input()

def figures_feedback_real_vs_belief():
    ie = False

    var_confidence_variable_true_noise_figure(0.1,interrupt_early=ie)
    var_confidence_variable_true_noise_figure(0.5,interrupt_early=ie)
    var_confidence_variable_true_noise_figure(1.0,interrupt_early=ie)
    var_confidence_variable_true_noise_figure(1.5,interrupt_early=ie)
    var_confidence_variable_true_noise_figure(3.0,interrupt_early=ie)
    # var_confidence_variable_true_noise_figure(5.0,interrupt_early=ie)
    input()

# A general function to create the recordings for a climb stair simulation
# with variable true_fb_std and belief_fb_std
def feedback_perception_simu(savepath,
                             true_feedback_std,belief_feedback_std,
                             Nsubj,Ntrials,action_selection_inverse_temp,
                            clamp_gaussian=False,override=False):
    
    T = 10
    Th = 2
    feedback_resolution = 5

    subj_cognitive_resolution = 5
    true_cognitive_resolution = 5

    pre_learnt_action_belief = 0.0
    initial_action_conf = 0.01

    a_prior_predominance = 100.0 
    initial_feedback_confidence = 10 # a0 = norm(ones + a_prior_predominance*gaussian_prior)*initial_feedback_confidence

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

    simulate_and_save(net,savepath,Nsubj,Ntrials,override=override)

def multiple_training_curves():
    # figures_feedback_real_vs_belief()
    # feedback_perception_simu(0.01,5.0)
    smooth_over = 5
    normalization_cst = 4.0
    Nsubj = 10
    Ntrials = 100

    action_selection_inverse_temp = 2.0
    true_fb_stds = [0.1,0.3,0.4,0.5,0.6,0.7,1.0]
    belief_feedback_stds = [0.1,0.5,1.0,1.5,3.0]
    ts = np.linspace(0,1.0,len(belief_feedback_stds))
    colorlist = [color_spectrum(np.array([1.0,0.0,0.0]),np.array([0.0,0.0,1.0]),t) for t in ts]
    
    def full_path(true_fb,belief_fb,asit):
        return os.path.join("simulation_outputs","paper1","stairs","belief_vs_true_fb_std",str(asit),"subject_expects_feedback_std_"+str(belief_fb)+"_noClamp","simulations_3."+str(true_fb)+".pickle")


    # Actually simulate & save the trials
    for true_fb_std in true_fb_stds:
        for belief_fb_std in belief_feedback_stds:
            feedback_perception_simu(full_path(true_fb_std,belief_fb_std,action_selection_inverse_temp),
                                     true_fb_std,belief_fb_std,
                                     Nsubj,Ntrials,action_selection_inverse_temp,
                                    clamp_gaussian=False,override=False)

    # Show the results
    fig,axes = plt.subplots(1,len(true_fb_stds),sharey=True)
    fig.suptitle("Subject training curve under variable subject prior feedback confidence and variable true feedback noise")

    fig2,axes2 = plt.subplots(1,len(true_fb_stds),sharey=True)
    fig2.suptitle("Subject training curve under variable subject prior feedback confidence and variable true feedback noise")
    
    for i,std in enumerate(true_fb_stds):
        savepaths = [full_path(std,k,action_selection_inverse_temp) for k in belief_feedback_stds]

        stms,weights,Nsubj,Ntrials = [],[],[],[]
        for belief_std,path,color in zip(belief_feedback_stds,savepaths,colorlist):
            _stm,_weight,_Nsubj,_Ntrials = extract_training_data(path)
            stms.append(_stm)
            weights.append(_weight)
            Nsubj = _Nsubj
            Ntrials = _Ntrials
        Xs = np.linspace(0,Ntrials,Ntrials)


        ax = axes[i]
        ax.set_xlabel("Trials")
        ax.grid()
        ax.set_ylim(0.0,1.0)
        ax.set_title("FB noise = " + str(std))
        
        ax2 = axes2[i]
        ax2.set_xlabel("Trials")
        ax2.grid()
        ax2.set_ylim(0.0,1.0)
        ax2.set_title("FB noise = " + str(std))

        # Plot smoothed training curves for all subjects : 
        # True observations :
        for stm,weight,color,label_curve in zip(stms,weights,colorlist,belief_feedback_stds):
            full_color = np.concatenate([color,np.array([1.0])],axis=0)
            trans_color = np.concatenate([color,np.array([0.2])],axis=0)

            observations = []
            mean_obs = []
            for subj in range(Nsubj):
                # print("________________________________________________________________________________________________________________________________")
                observations.append([])
                mean_obs.append([])
                feedback_learn_imgs = []
                for trial in range(1,Ntrials+1):
                    o_s = (stm[subj][trial][0].o)
                    u_s = (stm[subj][trial][1].u)

                    observations[-1].append(stm[subj][trial][1].o)
                    mean_obs[-1].append(np.mean(stm[subj][trial][1].o))
            smoothed_mean_obs = [clever_running_mean(subj_obs,smooth_over)/normalization_cst for subj_obs in mean_obs]
            smoothed_mean_arr = np.array(smoothed_mean_obs)

            for ksubj in range(Nsubj):
                ax.plot(Xs,smoothed_mean_arr[ksubj,:],color=full_color,linewidth=0.2)
            
            mean_values = np.mean(smoothed_mean_arr,axis=0) # mean of all subjects
            std_vals = np.std(smoothed_mean_arr,axis=0) # std of all subjects
            ax.plot(Xs,mean_values,color=full_color,linewidth=2.0,label=label_curve)
            ax.fill_between(Xs,mean_values-std_vals,mean_values+std_vals,color = trans_color)   
            
        # Plot smoothed state training curves for all subjects :
        # true cognitive level
        for stm,weight,color,label_curve in zip(stms,weights,colorlist,belief_feedback_stds):
            full_color = np.concatenate([color,np.array([1.0])],axis=0)
            trans_color = np.concatenate([color,np.array([0.2])],axis=0)

            all_states = []
            for subj in range(Nsubj):
                all_states.append([])
                for trial in range(1,Ntrials,1):
                    all_states[-1].append(np.mean(stm[subj][trial][0].x[0,:]))
            plot_smoothed_training_curves(all_states,ax2,label_curve,color,smooth_over,normalization_cst)

        if i ==(len(true_fb_stds)-1):
            ax.legend()
            ax2.legend()
        if i==0:
            ax.set_ylabel("Average feedback received")
            ax2.set_ylabel("Average cognitive lvl achieved")



    
    fig.show()
    fig2.show()
    input()

if __name__ == "__main__":
    multiple_training_curves()
    # figures_feedback_real_vs_belief()
    # feedback_perception_simu(0.01,5.0)

    # # Single + multiple_training_curves()
    # smooth_over = 2
    # normalization_cst = 4.0
    # Nsubj = 10
    # Ntrials = 100

    # def basepath(std_true,std_belief):
    #     return os.path.join("simulation_outputs","paper1","belief_vs_true_fb_std","subject_expects_feedback_std_"+str(std_belief)+"_noClamp","simulations_3."+str(std_true)+".pickle")

    # _stm,_weight,_Nsubj,_Ntrials = extract_training_data(basepath(0.5,0.5))
    #                                                             # TRUE/BELIEF
    # Xs = np.linspace(0,_Ntrials,_Ntrials)

    
    # # Plot smoothed training curves for all subjects : 
    # # True observations :
    # color = np.array([1.0,0.0,0.0])
    # color = np.array([0.0,0.0,1.0])
    # color = np.array([0.5,0.0,0.5])
    # full_color = np.concatenate([color,np.array([1.0])],axis=0)
    # trans_color = np.concatenate([color,np.array([0.2])],axis=0)


    # all_obs_back_to_back = []
    # observations = []
    # mean_obs = []
    # for subj in range(Nsubj):
    #     # print("________________________________________________________________________________________________________________________________")
    #     observations.append([])
    #     chain_of_obs = np.array([0.0])
        
    #     mean_obs.append([])
    #     feedback_learn_imgs = []
    #     for trial in range(1,Ntrials+1):
    #         o_s = (_stm[subj][trial][0].o)
    #         u_s = (_stm[subj][trial][1].u)

    #         observations[-1].append(_stm[subj][trial][1].o)
    #         mean_obs[-1].append(np.mean(_stm[subj][trial][1].o))
    #         # print(chain_of_obs)
    #         chain_of_obs = np.concatenate([chain_of_obs,o_s[0,:]])
    #     all_obs_back_to_back.append(chain_of_obs)
    # # print(np.array(all_obs_back_to_back[0]).shape)#/normalization_cst)

    # up_to_trial = 30*10
    # subj = 0
    # xs = np.linspace(0,chain_of_obs.shape[0],chain_of_obs.shape[0])
    # plt.scatter(xs[:up_to_trial],all_obs_back_to_back[subj][:up_to_trial]/normalization_cst,color=trans_color,s=2)
    # plt.plot(xs[:up_to_trial],clever_running_mean(all_obs_back_to_back[subj][:up_to_trial],10)/normalization_cst,color=full_color)
    # plt.show()


    # # exit()
    # fig,ax = plt.subplots(1)
    # smoothed_mean_obs = [clever_running_mean(subj_obs,smooth_over)/normalization_cst for subj_obs in mean_obs]
    # smoothed_mean_arr = np.array(smoothed_mean_obs)

    # for ksubj in range(Nsubj):
    #     ax.plot(Xs,smoothed_mean_arr[ksubj,:],color=full_color,linewidth=0.2)
    
    # mean_values = np.mean(smoothed_mean_arr,axis=0) # mean of all subjects
    # std_vals = np.std(smoothed_mean_arr,axis=0) # std of all subjects
    # ax.plot(Xs,mean_values,color=full_color,linewidth=2.0)
    # ax.fill_between(Xs,mean_values-std_vals,mean_values+std_vals,color = trans_color)   
    # ax.set_ylim(0.0,1.0)
    # fig.show()
    # input()