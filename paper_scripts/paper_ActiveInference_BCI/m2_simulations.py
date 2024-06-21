import numpy as np
import statistics as stat
import scipy.stats as scistats
import math,sys,os
import pickle 
import matplotlib.pyplot as plt

import actynf
from tools import clever_running_mean,color_spectrum
from paper_scripts.paper_ActiveInference_BCI.m2_model import neurofeedback_training_one_action

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

# A general function to create the recordings for a climb stair simulation
# with variable true_fb_std and belief_fb_std
def feedback_perception_simu(savepath,
                             true_feedback_std,belief_feedback_std,
                             Nsubj,Ntrials,generalization_temperature = 0.0,
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

    n_neutral_actions = 10 
    n_down_actions = 2
    n_up_actions = 1

    pLow = 0.5   # Without any increasing action, there is a pLow chance that the cognitive state will decrease spontaneously
    pUp  = 0.99

    
        # How much information s(t+1)=i|s(t)=j will tell us about
        #                    s(t+1)=i+k|s(t)=j+k (assuming linear state structure)


    net = neurofeedback_training_one_action(T,Th,  # Trial duration + temporal horizon
                    subj_cognitive_resolution,true_cognitive_resolution,       # Subject belief about cognitive resolution / true cognitive resolution
                    feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
                    belief_feedback_std,true_feedback_std,   # Subject belief about feedback noise / true feedback noise
                    n_up_actions,n_neutral_actions,n_down_actions,       # how many actions have no impact on the state ?
                    [pre_learnt_action_belief,initial_action_conf],  # Action mapping previous knowledge
                    [a_prior_predominance,initial_feedback_confidence],   # Feedback mapping previous knowledge
                    [d_prior_predominance,initial_d_confidence],   # d mapping previous knowledge
                    pLow,pUp,   # How likely it is that the cognitive state will go down when unattended
                                # / how likely it is that the correct action will increase the cognitive state
                    clamp_gaussian=clamp_gaussian,
                                # Weither to increase the categorical probabilistic weights
                                # on the edges or not
                    learning_space_structure=actynf.LINEAR,
                    gen_temp=generalization_temperature)
    simulate_and_save(net,savepath,Nsubj,Ntrials,override=override)

def full_sims():
    # figures_feedback_real_vs_belief()
    # feedback_perception_simu(0.01,5.0)
    smooth_over = 5
    normalization_cst = 4.0
    Nsubj = 10
    Ntrials = 100
    

    basepath = os.path.join("simulation_outputs","paper1","coherent_cognitive_space","belief_vs_true_fb_std")
    cold_color = np.array([0.0,0.0,1.0])
    hot_color = np.array([1.0,0.0,0.0])
    true_fb_stds = [0.1,0.3,0.4,0.5,0.6,0.7,1.0]
    belief_feedback_stds = [0.1,0.5,1.0,1.5,3.0]
    ts = np.linspace(0,1.0,len(belief_feedback_stds))
    colorlist = [color_spectrum(hot_color,cold_color,t) for t in ts]
    
    # Actually simulate & save the trials
    for true_fb_std in true_fb_stds:
        for belief_fb_std in belief_feedback_stds:
            feedback_perception_simu(true_fb_std,belief_fb_std,
                                     Nsubj,Ntrials,
                                    clamp_gaussian=False,override=False)

    # Show the results
    fig,axes = plt.subplots(1,len(true_fb_stds),sharey=True)
    fig.suptitle("Subject training curve under variable subject prior feedback confidence and variable true feedback noise")

    fig2,axes2 = plt.subplots(1,len(true_fb_stds),sharey=True)
    fig2.suptitle("Subject training curve under variable subject prior feedback confidence and variable true feedback noise")
    
    for i,std in enumerate(true_fb_stds):
        savepaths = [os.path.join(basepath(k),"simulations_3."+str(std)+".pickle") for k in belief_feedback_stds]

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

def total_generalization_sims():
    # figures_feedback_real_vs_belief()
    # feedback_perception_simu(0.01,5.0)
    smooth_over = 5
    normalization_cst = 4.0
    Nsubj = 10
    Ntrials = 50
    

    basepath = os.path.join("simulation_outputs","paper1","coherent_cognitive_space","belief_vs_true_fb_std")
    cold_color = np.array([0.0,0.0,1.0])
    hot_color = np.array([1.0,0.0,0.0])
    true_fb_stds = [0.1,0.3,0.5,0.7,1.0]
    belief_feedback_stds = [0.1,0.5,1.0]
    ts = np.linspace(0,1.0,len(belief_feedback_stds))
    colorlist = [color_spectrum(hot_color,cold_color,t) for t in ts]
    
    # Actually simulate & save the trials
    for true_fb_std in true_fb_stds:
        for belief_fb_std in belief_feedback_stds:
            svpth = os.path.join(basepath,str(true_fb_std)+"_"+str(belief_fb_std)+".pickle")
            feedback_perception_simu(svpth,
                                     true_fb_std,belief_fb_std,
                                     Nsubj,Ntrials,
                                    clamp_gaussian=False,override=False)
    
        # Show the results
    fig,axes = plt.subplots(1,len(true_fb_stds),sharey=True)
    fig.suptitle("Subject training curve under variable subject prior feedback confidence and variable true feedback noise")

    fig2,axes2 = plt.subplots(1,len(true_fb_stds),sharey=True)
    fig2.suptitle("Subject training curve under variable subject prior feedback confidence and variable true feedback noise")
    


    for i,std in enumerate(true_fb_stds):
        stms,weights,Nsubj,Ntrials = [],[],[],[]

        for belief_std,color in zip(belief_feedback_stds,colorlist):
            path = os.path.join(basepath,str(std)+"_"+str(belief_std)+".pickle")
            _stm,_weight,_Nsubj,_Ntrials = extract_training_data(path)
            stms.append(_stm)
            weights.append(_weight)
            # plt.imshow(_weight[0][30][0]["b"][0][:,:,0])
            # plt.show()
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
                for trial in range(1,Ntrials+1):
                    all_states[-1].append(np.mean(stm[subj][trial][0].x[0,:]))
            # plot_smoothed_training_curves(all_states,ax2,label_curve,color,smooth_over,normalization_cst)
            smoothed_mean_s = [clever_running_mean(state,smooth_over)/normalization_cst for state in all_states]
            smoothed_mean_arr = np.array(smoothed_mean_s)
            print(smoothed_mean_arr.shape)

            for ksubj in range(Nsubj):
                ax2.plot(Xs,smoothed_mean_arr[ksubj,:],color=full_color,linewidth=0.2)
            
            mean_values = np.mean(smoothed_mean_arr,axis=0) # mean of all subjects
            std_vals = np.std(smoothed_mean_arr,axis=0) # std of all subjects
            ax2.plot(Xs,mean_values,color=full_color,linewidth=2.0,label=label_curve)
            ax2.fill_between(Xs,mean_values-std_vals,mean_values+std_vals,color = trans_color)   
            

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
    # figures_feedback_real_vs_belief()
    # feedback_perception_simu(0.01,5.0)
    smooth_over = 5
    normalization_cst = 4.0
    Nsubj = 10
    Ntrials = 50

    generalization_temperature = 3.0
    

    basepath = os.path.join("simulation_outputs","paper1","coherent_cognitive_space","belief_vs_true_fb_std.gen=3.0")
    cold_color = np.array([0.0,0.0,1.0])
    hot_color = np.array([1.0,0.0,0.0])
    true_fb_stds = [0.1,0.3,0.5,0.7,1.0]
    belief_feedback_stds = [0.1,0.5,1.0]
    ts = np.linspace(0,1.0,len(belief_feedback_stds))
    colorlist = [color_spectrum(hot_color,cold_color,t) for t in ts]
    
    # Actually simulate & save the trials
    for true_fb_std in true_fb_stds:
        for belief_fb_std in belief_feedback_stds:
            svpth = os.path.join(basepath,str(true_fb_std)+"_"+str(belief_fb_std)+".pickle")
            feedback_perception_simu(svpth,
                                     true_fb_std,belief_fb_std,
                                     Nsubj,Ntrials,
                                     generalization_temperature=generalization_temperature,
                                    clamp_gaussian=False,override=False)
    
        # Show the results
    fig,axes = plt.subplots(1,len(true_fb_stds),sharey=True)
    fig.suptitle("Subject training curve under variable subject prior feedback confidence and variable true feedback noise")

    fig2,axes2 = plt.subplots(1,len(true_fb_stds),sharey=True)
    fig2.suptitle("Subject training curve under variable subject prior feedback confidence and variable true feedback noise")
    


    for i,std in enumerate(true_fb_stds):
        stms,weights,Nsubj,Ntrials = [],[],[],[]

        for belief_std,color in zip(belief_feedback_stds,colorlist):
            path = os.path.join(basepath,str(std)+"_"+str(belief_std)+".pickle")
            _stm,_weight,_Nsubj,_Ntrials = extract_training_data(path)
            stms.append(_stm)
            weights.append(_weight)
            # plt.imshow(_weight[0][30][0]["b"][0][:,:,0])
            # plt.show()
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
                for trial in range(1,Ntrials+1):
                    all_states[-1].append(np.mean(stm[subj][trial][0].x[0,:]))
            # plot_smoothed_training_curves(all_states,ax2,label_curve,color,smooth_over,normalization_cst)
            smoothed_mean_s = [clever_running_mean(state,smooth_over)/normalization_cst for state in all_states]
            smoothed_mean_arr = np.array(smoothed_mean_s)
            print(smoothed_mean_arr.shape)

            for ksubj in range(Nsubj):
                ax2.plot(Xs,smoothed_mean_arr[ksubj,:],color=full_color,linewidth=0.2)
            
            mean_values = np.mean(smoothed_mean_arr,axis=0) # mean of all subjects
            std_vals = np.std(smoothed_mean_arr,axis=0) # std of all subjects
            ax2.plot(Xs,mean_values,color=full_color,linewidth=2.0,label=label_curve)
            ax2.fill_between(Xs,mean_values-std_vals,mean_values+std_vals,color = trans_color)   
            
            for trialn in range(Ntrials):
                print("---")
                print(np.round(weight[0][trialn][1]["b"][0][:,:,0],2))

        if i ==(len(true_fb_stds)-1):
            ax.legend()
            ax2.legend()
        if i==0:
            ax.set_ylabel("Average feedback received")
            ax2.set_ylabel("Average cognitive lvl achieved")
    fig.show()
    fig2.show()
    input()