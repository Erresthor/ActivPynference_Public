import numpy as np
import statistics as stat
import scipy.stats as scistats
import math,sys,os
import pickle 
import matplotlib.pyplot as plt

import actynf
from tools import clever_running_mean,color_spectrum
from m1_IO_model import neurofeedback_training

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

# A general function to create the recordings for a climb stair simulation
# with variable true_fb_std and belief_fb_std
def feedback_perception_simu(savepath,
                             true_feedback_std,belief_feedback_std,
                             Nsubj,Ntrials,action_selection_inverse_temp,
                            clamp_gaussian=False,override=False):
    learn_a = True

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
            belief_feedback_std,true_feedback_std,   # Subject belief about feedback noise / true feedback noise
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

    simulate_and_save(net,savepath,Nsubj,Ntrials,override=override)

def multiple_training_curves():
    # figures_feedback_real_vs_belief()
    # feedback_perception_simu(0.01,5.0)
    smooth_over = 5
    normalization_cst = 4.0
    Nsubj = 10
    Ntrials = 100

    action_selection_inverse_temp = 32.0
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

    true_feedback_std = 1.0 #2.0
    true_intero_std = 0.01
    belief_feedback_std = 0.25
    Nsubj = 10
    Ntrials = 100
    action_selection_inverse_temp = 32.0
    learn_a = True
    override = False

    savepath = os.path.join(
        "simulation_outputs",
        "paper1",
        "stairs",
        "IO_simulations",
        "true_FB_"+str(true_feedback_std),
        "belief_FB_" + str(belief_feedback_std),
        "true_intero_"+str(true_intero_std)+".pickle")


    T = 10
    Th = 2
    feedback_resolution = 5

    subj_cognitive_resolution = 5
    true_cognitive_resolution = 5

    k1b = 0.01
    epsilon_b = 0.01

    k1a1 = 10
    epsilon_a = 1.0/101.0 # a0 = norm((1/101)* ones + gaussian_prior)*k1a
    k1a2 = 10

    k1d = 1.0
    epsilon_d = 1.0

    neutral_action_prop = 0.2 # 20% of the actions have no interest for the task

    pLow = 0.5   # Without any increasing action, there is a pLow chance that the cognitive state will decrease spontaneously
    pUp  = 0.99


    net = neurofeedback_training(T,Th,  # Trial duration + temporal horizon
            subj_cognitive_resolution,true_cognitive_resolution,       # Subject belief about cognitive resolution / true cognitive resolution
            feedback_resolution,feedback_resolution,true_intero_std,       # Subject belief about feedback resolution / true feedback resolution
            belief_feedback_std,true_feedback_std,   # Subject belief about feedback noise / true feedback noise
            neutral_action_prop,       # how many actions have no impact on the state ?
            k1b,epsilon_b,  # Action mapping previous knowledge
            k1a1,epsilon_a,k1a2,   # Feedback mapping previous knowledge
            k1d,epsilon_d,   # d mapping previous knowledge
            pLow,pUp,   # How likely it is that the cognitive state will go down when unattended
                        # / how likely it is that the correct action will increase the cognitive state
            asit = action_selection_inverse_temp,
            learn_a=learn_a) 
                        # Clamp : Weither to increase the categorical probabilistic weights
                        # on the edges or not
                        # asit : inverse temperature of the action selection process
                        # learn_a : Weither to learn the perception matrix on the go                                       

    simulate_and_save(net,savepath,Nsubj,Ntrials,override=override)

    _stm,_weight,_Nsubj,_Ntrials = extract_training_data(savepath)

    def init():
        im.set_data(np.random.random((5,5)))
        return [im]

    # animation function.  This is called sequentially
    def animate(i):
        a=im.get_array()
        a=a*np.exp(-0.001*i)    # exponential decay of the values
        im.set_array(a)
        return [im]

    for subj in range(1):
        imlist = []
        for trial in range(Ntrials):
            imlist.append(actynf.normalize(_weight[subj][trial][1]["a"][1]))
            print(_weight[subj][trial][1]["a"][1])
    
    import matplotlib.animation as animation
    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(imlist[0],vmin=0.0,vmax=1.0)

    def init():
        im.set_data(imlist[0])
        return im

    def animate(i):
        a=imlist[i]
        im.set_array(a)
        return [im]
    
    ani = animation.FuncAnimation(fig, animate, frames=range(Ntrials), 
                              interval=50, blit=True)
    plt.show()

    # multiple_training_curves()
    # figures_feedback_real_vs_belief()
    # feedback_perception_simu(0.01,5.0)

    # # Single + multiple_training_curves()
    # smooth_over = 2
    # normalization_cst = 4.0
    # Nsubj = 10
    # Ntrials = 100

    # def basepath(std_true,std_belief):
    #     return os.path.join("simulation_outputs","paper1","belief_vs_true_fb_std","subject_expects_feedback_std_"+str(std_belief)+"_noClamp","simulations_3."+str(std_true)+".pickle")

    
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