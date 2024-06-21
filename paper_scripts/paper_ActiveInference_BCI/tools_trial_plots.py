import sys,os
import matplotlib.pyplot as plt
import numpy as np
import actynf

from tools import dist_kl_dir,js_dir

def colorfunc(colorlist,t,interp = 'linear'):
    n = len(colorlist)
    if (interp=='linear'):
        for i in range(n):
            current_color_prop = (float(i)/(n - 1))
            next_color_prop = (float(i+1)/(n-1))
            if ((t>=current_color_prop) and (t<=next_color_prop)):
                ti = (t - current_color_prop)/(next_color_prop-current_color_prop)
                return colorlist[i+1]*ti + colorlist[i]*(1-ti)

def custom_colormap(colormap,in_array,interpolation='linear') :
    """Not very elegant + only designed for 3D matrices :>(  """
    output_array = np.zeros(in_array.shape+colormap[0].shape)
    for x in range(in_array.shape[0]):
        for y in range(in_array.shape[1]):
            output_array[x,y,:] = colorfunc(colormap,in_array[x,y],interp=interpolation)
    return output_array

def trial_plot(observations,
                      real_states,states_beliefs,
                      actions,action_beliefs) :
    titlesize = 8

    T = observations.shape[0]

    n_actions = action_beliefs.shape[0]
    timesteps = np.linspace(0,T-1,T)

    my_colormap= [np.array([80,80,80,200]) , np.array([39,136,245,200]) , np.array([132,245,39,200]) , np.array([245,169,39,200]) , np.array([255,35,35,200])]
    my_colormap_bw = [np.array([255,255,255,255]) , np.array([0,0,0,255])]

    N = 250
    img_array = np.linspace(1,0,N)
    img = np.zeros(img_array.shape +(50,) +  (4,))
    for k in range(N):
        color_array = colorfunc(my_colormap,img_array[k])
        img[k,:,:] = color_array
    
    state_belief_image = custom_colormap(my_colormap_bw,states_beliefs)
    action_belief_image = custom_colormap(my_colormap_bw,action_beliefs)   

    # Major ticks every 5, minor ticks every 1
    minor_ticks_x = np.arange(0, T, 1)
    ticks_y =  np.arange(0, 5, 1)
    ticks_y_long = np.arange(0, n_actions, 1)

    # BEGIN ! --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    fig = plt.figure()
    axes = fig.subplots(3,1,sharex=True)

    ax1 = axes[0]

    labels = [str(i) for i in minor_ticks_x]
    ax1.set_xticks(minor_ticks_x)
    ax1.set_xticklabels(labels)

    ax1.set_yticks(ticks_y)
    ax1.grid()
    ax1.set_title("Feedback observations",fontsize=titlesize)
    ax1.plot(timesteps,observations,color = 'darkorange',marker="H",linestyle = 'None',markersize=10,label="Observations")
    ax1.set_ylim([-0.5,4.5])
    ax1.set_ylabel("Observations")

    ax2 = axes[1]
    ax2.set_title("Subject belief about corresponding mental state",fontsize=titlesize)
    ax2.set_yticks(ticks_y_long)
    ax2.set_ylim([-0.5,4.5])
    ax2.set_ylabel("States")
    ax2.set(xlim=(0-0.5, T-0.5))
    ax2.imshow(state_belief_image/255.0,aspect="auto")
    ax2.plot(timesteps,real_states,color='red',marker="p",linestyle = 'None',markersize=8,label="True state")
    ax2.grid()
    ax2.legend()

    ax3 = axes[2]
    ax3.set_title("Action selection",fontsize=titlesize)
    ax3.set_yticks(ticks_y_long)
    ax3.set_ylabel("Actions")
    ax3.set_ylim([-0.5,n_actions - 0.5])
    ax3.set(xlim=(0-0.5, T-0.5))
    ax3.imshow(action_belief_image/255.0,aspect="auto")
    ax3.plot(timesteps[:-1],actions,color='blue',marker="H",linestyle = 'None',markersize=10,label="Selected action")
    ax3.grid()
    ax3.set_xlabel("Timesteps")
    ax3.legend()

    return fig,axes

def trial_plot_figure(fig,observations,obs_dist,
                      real_states,states_beliefs,
                      actions,action_beliefs) :
    titlesize = 8

    T = observations.shape[0]

    n_actions = action_beliefs.shape[0]
    timesteps = np.linspace(0,T-1,T)

    my_colormap= [np.array([80,80,80,200]) , np.array([39,136,245,200]) , np.array([132,245,39,200]) , np.array([245,169,39,200]) , np.array([255,35,35,200])]
    my_colormap_bw = [np.array([255,255,255,255]) , np.array([0,0,0,255])]

    N = 250
    img_array = np.linspace(1,0,N)
    img = np.zeros(img_array.shape +(50,) +  (4,))
    for k in range(N):
        color_array = colorfunc(my_colormap,img_array[k])
        img[k,:,:] = color_array
    

    if obs_dist.ndim >2:
        obs_dist = np.sum(obs_dist,axis=1)
        # obs_dist = obs_dist[]
    obs_dist_image = custom_colormap(my_colormap_bw,obs_dist)
    state_belief_image = custom_colormap(my_colormap_bw,states_beliefs)
    action_belief_image = custom_colormap(my_colormap_bw,action_beliefs)   

    # Major ticks every 5, minor ticks every 1
    minor_ticks_x = np.arange(0, T, 1)
    ticks_y =  np.arange(0, 5, 1)
    ticks_y_long = np.arange(0, n_actions, 1)

    # BEGIN ! --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    axes = fig.subplots(3,1,sharex=True)

    ax1 = axes[0]

    labels = [str(i) for i in minor_ticks_x]
    ax1.set_xticks(minor_ticks_x)
    ax1.set_xticklabels(labels)

    ax1.set_yticks(ticks_y)
    ax1.grid()
    ax1.set_title("Feedback observations",fontsize=titlesize)
    ax1.imshow(obs_dist_image/255.0,aspect="auto")
    ax1.plot(timesteps,observations,color = 'darkgreen',marker="H",linestyle = 'None',markersize=10,label="Feedback lvl")
    ax1.set_ylim([-0.5,4.5])
    ax1.set_ylabel("Observations")
    ax1.legend()

    ax2 = axes[1]
    ax2.set_title("Subject belief about corresponding mental state",fontsize=titlesize)
    ax2.set_yticks(ticks_y_long)
    ax2.set_ylim([-0.5,4.5])
    ax2.set_ylabel("States")
    ax2.set(xlim=(0-0.5, T-0.5))
    ax2.imshow(state_belief_image/255.0,aspect="auto")
    ax2.plot(timesteps,real_states,color='red',marker="p",linestyle = 'None',markersize=8,label="True state")
    ax2.grid()
    ax2.legend()

    ax3 = axes[2]
    ax3.set_title("Action selection",fontsize=titlesize)
    ax3.set_yticks(ticks_y_long)
    ax3.set_ylabel("Actions")
    ax3.set_ylim([-0.5,n_actions - 0.5])
    ax3.set(xlim=(0-0.5, T-0.5))
    ax3.imshow(action_belief_image/255.0,aspect="auto")
    ax3.plot(timesteps[:-1],actions,color='blue',marker="H",linestyle = 'None',markersize=10,label="Selected action")
    ax3.grid()
    ax3.set_xlabel("Timesteps")
    ax3.legend()

    return fig,axes

def plot_one_trial(stm,weight,
        subject_id,trial_id,title,
        plot_true_matrices=False,
        save_fig_path=None,name=""):
    
    bigfig = plt.figure(figsize=(6,5), constrained_layout=True)
    if plot_true_matrices:
        subfigs = bigfig.subfigures(1, 3, width_ratios=[5,1,1], wspace=0.05)
    else : 
        subfigs = bigfig.subfigures(1, 2, width_ratios=[5,1], wspace=0.05)
    trial_id = trial_id + 1 # Trial 0 does not exist !
    stm_sub_trial = stm[subject_id][trial_id]
    
    true_mental_states = stm_sub_trial[0].x[0]
    feedback_levels = stm_sub_trial[0].o[0]
    feedback_distribution = stm_sub_trial[0].o_d

    subject_state_inferences = stm_sub_trial[1].x_d
    subject_mental_actions = stm_sub_trial[1].u
    subject_action_posteriors = stm_sub_trial[1].u_d

    # subfigs[0] = trial_plot_figure(feedback_levels,
    #                   true_mental_states,subject_state_inferences,
    #                   subject_mental_actions,subject_action_posteriors)
    trial_fig,trial_axes = trial_plot_figure(subfigs[0],feedback_levels,feedback_distribution,
                      true_mental_states,subject_state_inferences,
                      subject_mental_actions,subject_action_posteriors)
    subfigs[0].suptitle(title)
    
    
    weights_sub_trial = actynf.normalize(weight[subject_id][trial_id-1][1]["b"][0])
    axes = subfigs[1].subplots(weights_sub_trial.shape[-1])
    for k,ax in enumerate(axes):
        ax.imshow(weights_sub_trial[:,:,k],vmin=0,vmax=1)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_title("Action " + str(k))
    subfigs[1].suptitle("Model b")

    if plot_true_matrices:
        weights_sub_trial = actynf.normalize(weight[subject_id][trial_id-1][0]["b"][0])
        axes = subfigs[2].subplots(weights_sub_trial.shape[-1])
        for k,ax in enumerate(axes):
            ax.imshow(weights_sub_trial[:,:,k],vmin=0,vmax=1)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_title("Action " + str(k))
        subfigs[2].suptitle("Process B")
        bigfig.show()

    if not(save_fig_path == None):
        if not os.path.exists(save_fig_path):
            os.makedirs(save_fig_path)
        bigfig.savefig(os.path.join(save_fig_path,name))
    return bigfig

def plot_training_curve(stm,weights,title,save_fig_path=None,name="",
                        distance_metric="kl"):
    fig,axes = plt.subplots(2,2,dpi=200,sharex=True)#,sharey=True)

    smooth_over = 2
    # Show the results

    Nsubj = len(stm)
    Ntrials = len(stm[0])
    
    process_state_array = np.array([[stm[sub_id][trial_id][0].x[0] for trial_id in range(1,Ntrials,1)] for sub_id in range(Nsubj)])
    process_obs_array = np.array([[stm[sub_id][trial_id][0].o[0] for trial_id in range(1,Ntrials,1)] for sub_id in range(Nsubj)])

    subject_state_array = np.array([[stm[sub_id][trial_id][1].x_d for trial_id in range(1,Ntrials,1)] for sub_id in range(Nsubj)])
    subject_action_array = np.array([[stm[sub_id][trial_id][0].u for trial_id in range(1,Ntrials,1)] for sub_id in range(Nsubj)])
    subject_action_posterior = np.array([[stm[sub_id][trial_id][0].u_d for trial_id in range(1,Ntrials,1)] for sub_id in range(Nsubj)])
    
    Ns = subject_state_array.shape[2]

    XTrials = np.linspace(1,Ntrials,Ntrials-1)
    normalization_cst = Ns-1 # To get 

    fig.suptitle(title,fontsize=10)

    # Plot the average level of feedback received by the subjects
    axes[0,0].set_title("Feedback levels")
    avg_fb = np.mean(process_obs_array,axis = -1)#/normalization_cst
    avg_fb_allsubjs = np.mean(avg_fb,axis=0)
    var_fb_allsubjs = np.std(avg_fb,axis=0)
    for subj in range(Nsubj):
        axes[0,0].fill_between(XTrials,avg_fb_allsubjs-var_fb_allsubjs,avg_fb_allsubjs+var_fb_allsubjs) 
        axes[0,0].plot(XTrials,avg_fb[subj,:],color="black",linewidth=0.2)
    axes[0,0].plot(XTrials,avg_fb_allsubjs,color="black",linewidth=2.0,label="States")
    # axes[0,0].set_ylim([0.0,1.0])
    axes[0,0].set_ylim([0.0,normalization_cst])
    axes[0,0].grid()

    # Plot the average cognitive states achieved by the subjects
    axes[0,1].set_title("Mental states")
    avg_state = np.mean(process_state_array,axis = -1)#/normalization_cst
    avg_state_allsubjs = np.mean(avg_state,axis=0)
    var_state_allsubjs = np.std(avg_state,axis=0)
    for subj in range(Nsubj):
        axes[0,1].fill_between(XTrials,avg_state_allsubjs-var_state_allsubjs,avg_state_allsubjs+var_state_allsubjs) 
        axes[0,1].plot(XTrials,avg_state[subj,:],color="black",linewidth=0.2)
    axes[0,1].plot(XTrials,avg_state_allsubjs,color="black",linewidth=2.0,label="States")
    # axes[0,1].set_ylim([0.0,1.0])
    axes[0,1].set_ylim([0.0,normalization_cst])
    axes[0,1].grid()

    subj_fixed = 0
    true_A = weights[subj_fixed][0][0]["a"][0]
    true_B = weights[subj_fixed][0][0]["b"][0]

    if distance_metric.lower()=="js":
        a_dist_array = np.array([[js_dir(weights[subj][trial][1]["a"][0],true_A) for trial in range(Ntrials)] for subj in range(Nsubj)])
        b_dist_array = np.array([[js_dir(weights[subj][trial][1]["b"][0],true_B) for trial in range(Ntrials)] for subj in range(Nsubj)])
        axes[1,0].set_ylabel("$JS_{dir}[\mathbf{b}_{model},\mathbf{B}_{process}]$",fontsize=8)
        axes[1,1].set_ylabel("$JS_{dir}[\mathbf{a}_{model},\mathbf{A}_{process}]$",fontsize=8)
    elif distance_metric.lower()=="kl":
        a_dist_array = np.array([[dist_kl_dir(weights[subj][trial][1]["a"][0],true_A) for trial in range(Ntrials)] for subj in range(Nsubj)])
        b_dist_array = np.array([[dist_kl_dir(weights[subj][trial][1]["b"][0],true_B) for trial in range(Ntrials)] for subj in range(Nsubj)])
        axes[1,0].set_ylabel("$KL_{dir}[\mathbf{b}_{model},\mathbf{B}_{process}]$",fontsize=8)
        axes[1,1].set_ylabel("$KL_{dir}[\mathbf{a}_{model},\mathbf{A}_{process}]$",fontsize=8)
    else : 
        raise ValueError("Argument not recognized for distribution metric : " + distance_metric + " -- should be either kl (Kullback-Leibler) or js (Jensen-Shannon)")

    XTrialsP1 = np.linspace(0,Ntrials,Ntrials)

    # Plot the evolution of the mental action model developped by the subjects
    axes[1,0].set_title("Action model error")
    avg_b_err_allsubjs = np.mean(b_dist_array,axis=0)
    var_b_err_allsubjs = np.std(b_dist_array,axis=0)
    for subj in range(Nsubj):
        axes[1,0].fill_between(XTrialsP1,avg_b_err_allsubjs-var_b_err_allsubjs,avg_b_err_allsubjs+var_b_err_allsubjs) 
        axes[1,0].plot(XTrialsP1,b_dist_array[subj,:],color="black",linewidth=0.2)
    axes[1,0].plot(XTrialsP1,avg_b_err_allsubjs,color="black",linewidth=2.0,label="States")
    axes[1,0].set_ylim(bottom=0)
    axes[1,0].grid()

    # Plot the evolution of the mental action model developped by the subjects
    axes[1,1].set_title("Feedback model error")
    avg_a_err_allsubjs = np.mean(a_dist_array,axis=0)
    var_a_err_allsubjs = np.std(a_dist_array,axis=0)
    for subj in range(Nsubj):
        axes[1,1].fill_between(XTrialsP1,avg_a_err_allsubjs-var_a_err_allsubjs,avg_a_err_allsubjs+var_a_err_allsubjs) 
        axes[1,1].plot(XTrialsP1,a_dist_array[subj,:],color="black",linewidth=0.2)
    axes[1,1].plot(XTrialsP1,avg_a_err_allsubjs,color="black",linewidth=2.0,label="States")
    axes[1,1].set_ylim(bottom=0)
    if (np.max(avg_a_err_allsubjs)<1e-10):
        axes[1,1].set_ylim([0,1])
    axes[1,1].grid()

    axes[1,0].set_xlabel("Trials",fontsize=8)
    axes[1,1].set_xlabel("Trials",fontsize=8)

    axes[0,0].set_ylabel("Avg. mental state",fontsize=8)
    axes[0,1].set_ylabel("Avg. feedback level",fontsize=8)
    # axes[1,0].set_ylabel("KL[model,true]",fontsize=8)
    if not(save_fig_path == None):
        if not os.path.exists(save_fig_path):
            os.makedirs(save_fig_path)
        fig.savefig(os.path.join(save_fig_path,name))
    return fig


def plot_overlayed_training_curve(stmlist,weightslist,labellist,
                                  title,
                                  save_fig_path=None,name="",
                                  colorlist=None,
                                  distance_metric="kl"):
    fig,axes = plt.subplots(2,2,dpi=200,sharex=True)#,sharey=True)
    fig.suptitle(title,fontsize=10)
    
    Nsubj = len(stmlist[0])
    Ntrials = len(stmlist[0][0])
    subject_state_array = np.array([[stmlist[0][sub_id][trial_id][1].x_d for trial_id in range(1,Ntrials,1)] for sub_id in range(Nsubj)])
    Ns = subject_state_array.shape[2]

    XTrials = np.linspace(1,Ntrials,Ntrials-1)
    XTrialsP1 = np.linspace(0,Ntrials,Ntrials)
    
    normalization_cst = Ns-1 # To get 

    smooth_over = 2
    
    if colorlist==None:
        colorlist = [np.random.random((3,)) for stm in stmlist]
    
    # Show the results
    for stm,weights,color,label in zip(stmlist,weightslist,colorlist,labellist):
        linecolor = color
        fillcolor = color
        
        process_state_array = np.array([[stm[sub_id][trial_id][0].x[0] for trial_id in range(1,Ntrials,1)] for sub_id in range(Nsubj)])
        process_obs_array = np.array([[stm[sub_id][trial_id][0].o[0] for trial_id in range(1,Ntrials,1)] for sub_id in range(Nsubj)])

        subject_state_array = np.array([[stm[sub_id][trial_id][1].x_d for trial_id in range(1,Ntrials,1)] for sub_id in range(Nsubj)])
        subject_action_array = np.array([[stm[sub_id][trial_id][0].u for trial_id in range(1,Ntrials,1)] for sub_id in range(Nsubj)])
        subject_action_posterior = np.array([[stm[sub_id][trial_id][0].u_d for trial_id in range(1,Ntrials,1)] for sub_id in range(Nsubj)])
        
        # Plot the average level of feedback received by the subjects
        avg_fb = np.mean(process_obs_array,axis = -1)#/normalization_cst
        avg_fb_allsubjs = np.mean(avg_fb,axis=0)
        var_fb_allsubjs = np.std(avg_fb,axis=0)
        axes[0,0].fill_between(XTrials,avg_fb_allsubjs-var_fb_allsubjs,avg_fb_allsubjs+var_fb_allsubjs,color=fillcolor,alpha=0.1) 
        # for subj in range(Nsubj):
        #     axes[0,0].plot(XTrials,avg_fb[subj,:],color=linecolor,linewidth=0.2)
        axes[0,0].plot(XTrials,avg_fb_allsubjs,color=linecolor,linewidth=2.0,alpha=0.5)
        # axes[0,0].set_ylim([0.0,1.0])

        # Plot the average cognitive states achieved by the subjects
        avg_state = np.mean(process_state_array,axis = -1)#/normalization_cst
        avg_state_allsubjs = np.mean(avg_state,axis=0)
        var_state_allsubjs = np.std(avg_state,axis=0)
        axes[0,1].fill_between(XTrials,avg_state_allsubjs-var_state_allsubjs,avg_state_allsubjs+var_state_allsubjs,color=fillcolor,alpha=0.1) 
        # for subj in range(Nsubj):
        #     axes[0,1].plot(XTrials,avg_state[subj,:],color=linecolor,linewidth=0.2)
        axes[0,1].plot(XTrials,avg_state_allsubjs,color=linecolor,linewidth=2.0,alpha=0.5)

        subj_fixed = 0
        true_A = weights[subj_fixed][0][0]["a"][0]
        true_B = weights[subj_fixed][0][0]["b"][0]

        a_dist_array = np.array([[dist_kl_dir(weights[subj][trial][1]["a"][0],true_A) for trial in range(Ntrials)] for subj in range(Nsubj)])
        b_dist_array = np.array([[dist_kl_dir(weights[subj][trial][1]["b"][0],true_B) for trial in range(Ntrials)] for subj in range(Nsubj)])

        if distance_metric.lower()=="js":
            a_dist_array = np.array([[js_dir(weights[subj][trial][1]["a"][0],true_A) for trial in range(Ntrials)] for subj in range(Nsubj)])
            b_dist_array = np.array([[js_dir(weights[subj][trial][1]["b"][0],true_B) for trial in range(Ntrials)] for subj in range(Nsubj)])
            axes[1,0].set_ylabel("$JS_{dir}[\mathbf{b}_{model},\mathbf{B}_{process}]$",fontsize=8)
            axes[1,1].set_ylabel("$JS_{dir}[\mathbf{a}_{model},\mathbf{A}_{process}]$",fontsize=8)
        elif distance_metric.lower()=="kl":
            a_dist_array = np.array([[dist_kl_dir(weights[subj][trial][1]["a"][0],true_A) for trial in range(Ntrials)] for subj in range(Nsubj)])
            b_dist_array = np.array([[dist_kl_dir(weights[subj][trial][1]["b"][0],true_B) for trial in range(Ntrials)] for subj in range(Nsubj)])
            axes[1,0].set_ylabel("$KL_{dir}[\mathbf{b}_{model},\mathbf{B}_{process}]$",fontsize=8)
            axes[1,1].set_ylabel("$KL_{dir}[\mathbf{a}_{model},\mathbf{A}_{process}]$",fontsize=8)
        else : 
            raise ValueError("Argument not recognized for distribution metric : " + distance_metric + " -- should be either kl (Kullback-Leibler) or js (Jensen-Shannon)")

        # Plot the evolution of the mental action model developped by the subjects
        avg_b_err_allsubjs = np.mean(b_dist_array,axis=0)
        var_b_err_allsubjs = np.std(b_dist_array,axis=0)

        axes[1,0].fill_between(XTrialsP1,avg_b_err_allsubjs-var_b_err_allsubjs,avg_b_err_allsubjs+var_b_err_allsubjs,color=fillcolor,alpha=0.1) 
        # for subj in range(Nsubj):
        #     axes[1,0].plot(XTrialsP1,b_dist_array[subj,:],color=linecolor,linewidth=0.2)
        axes[1,0].plot(XTrialsP1,avg_b_err_allsubjs,color=linecolor,linewidth=2.0,alpha=0.5)
        

        # Plot the evolution of the mental action model developped by the subjects
        
        avg_a_err_allsubjs = np.mean(a_dist_array,axis=0)
        var_a_err_allsubjs = np.std(a_dist_array,axis=0)
        axes[1,1].fill_between(XTrialsP1,avg_a_err_allsubjs-var_a_err_allsubjs,avg_a_err_allsubjs+var_a_err_allsubjs,color=fillcolor,alpha=0.1) 
        # for subj in range(Nsubj):
        #     axes[1,1].plot(XTrialsP1,a_dist_array[subj,:],color=linecolor,linewidth=0.2)

        axes[1,1].plot(XTrialsP1,avg_a_err_allsubjs,color=linecolor,linewidth=2.0,label=label,alpha=0.5)
    
    axes[0,0].set_title("Feedback levels")
    axes[0,0].set_ylim([0.0,normalization_cst])
    axes[0,0].grid()

    axes[0,1].set_title("Mental states")
    axes[0,1].set_ylim([0.0,normalization_cst])
    axes[0,1].grid()

    axes[1,0].set_title("Action model error")
    axes[1,0].set_ylim(bottom=0)
    axes[1,0].grid()
    
    axes[1,1].set_title("Feedback model error")
    axes[1,1].set_ylim(bottom=0)
    axes[1,1].grid()
    axes[1,1].legend(loc="best")

    axes[1,0].set_xlabel("Trials",fontsize=8)
    axes[1,1].set_xlabel("Trials",fontsize=8)
    axes[0,0].set_ylabel("Avg. mental state",fontsize=8)
    axes[0,1].set_ylabel("Avg. feedback level",fontsize=8)
    
    if not(save_fig_path == None):
        if not os.path.exists(save_fig_path):
            os.makedirs(save_fig_path)
        fig.savefig(os.path.join(save_fig_path,name))
    return fig

def colormap_plot_2D(simulation_parameters,
                     s_perf,a_perf,b_perf,
                  last_K_trials = 15,title="",interoceptive_plot=False,
                  save_fig_path=None,name="",showtitle=False,max_state=3.5):
    
    a_perf[np.isnan(a_perf)] = 0.0

    array_of_final_states = np.mean(s_perf[:,:,:,-last_K_trials:,:],axis = (-1,-2,-3))
    array_of_final_a = np.mean(a_perf[:,:,:,-1],axis = (-1))
    array_of_final_b = np.mean(b_perf[:,:,:,-1],axis = (-1))
    max_b = np.max(b_perf[...,0])
    min_b = np.min(array_of_final_b)

    # getting the original colormap using cm.get_cmap() function 
    orig_map=plt.cm.get_cmap('viridis') 
    reversed_map = orig_map.reversed() 

    fig2,axes = plt.subplots(1,2,sharey=True,dpi=100)
    if(showtitle):
        fig2.suptitle(title,fontsize = 10)
    # fig2.tight_layout()
    # fig2.subplots_adjust(top=0.95)

    s_dist=axes[0].imshow(array_of_final_states,vmin=0,interpolation='nearest',vmax=max_state)
    axes[0].set_title("Avg. cognitive state \n (last 15 trials)",fontsize = 8)
    axes[0].set_ylabel("$\sigma_{process}$",fontsize = 8)
    axes[0].set_xlabel("$\sigma_{model}$",fontsize = 8)

    b_dist=axes[1].imshow(array_of_final_b,interpolation='nearest',cmap = reversed_map,
                                vmin=0,vmax=max_b)
    axes[1].set_title("Transition model error (after training)",fontsize = 8)
    axes[1].set_xlabel("$\sigma_{model}$",fontsize = 8)

    x_axis = simulation_parameters[:,0,0]
    y_axis = simulation_parameters[0,:,1]
    for ax in axes :
        ax.set_yticks(range(x_axis.shape[0]))
        ax.set_yticklabels(np.round(x_axis,1),fontsize = 3)

        ax.set_xticks(range(y_axis.shape[0]))
        ax.set_xticklabels(np.round(y_axis,1),fontsize = 3)
    ax.invert_yaxis()

    fig2.tight_layout()
    fig2.colorbar(s_dist, ax=axes[0],shrink=0.5)
    fig2.colorbar(b_dist, ax=axes[1],shrink=0.5)

    if not(save_fig_path == None):
        if not os.path.exists(save_fig_path):
            os.makedirs(save_fig_path)
        fig2.savefig(os.path.join(save_fig_path,name),bbox_inches='tight')
    return fig2

def diff_2D_colormap(simulation_performances_no_learn,simulation_performances_learn,idx,savetofolder):
    # idx = 9
    fig,axs = plt.subplots(1,2)
    fig.tight_layout()
    lastK_trials = 15
    s_perf_dif = np.mean(simulation_performances_no_learn["s"][:,:,:,-lastK_trials:,:],axis=(-1,-2)) - np.mean(simulation_performances_learn["s"][:,:,idx,:,-lastK_trials:,:],axis=(-1,-2))
    absmax = np.max(np.abs(np.mean(s_perf_dif,axis=-1)))
    pcm = axs[0].imshow(np.mean(s_perf_dif,axis=-1),cmap="bwr",vmin=-absmax,vmax=absmax)
    axs[0].invert_yaxis()
    axs[0].set_title("Final mental state \n no learning  -  learning (k1a="+str(idx+1.0)+")",fontsize = 9)
    axs[0].set_xticks(range(0,20,19),np.round(np.linspace(0.01,2.0,2),1),fontsize=9)
    axs[0].set_yticks(range(0,20,19),np.round(np.linspace(0.01,2.0,2),1),fontsize=9)
    fig.colorbar(pcm, ax=axs[0],shrink=0.5)

    b_perf_dif = simulation_performances_no_learn["b"][:,:,:,-1] - simulation_performances_learn["b"][:,:,idx,:,-1]
    absmax = np.max(np.abs(np.mean(b_perf_dif,axis=-1)))
    pcm = axs[1].imshow(np.mean(b_perf_dif,axis=-1),cmap="bwr_r",vmax=-absmax,vmin=absmax)
    axs[1].set_title("Final action model error \n no learning  -  learning(k1a="+str(idx+1.0)+")",fontsize = 9)
    axs[1].set_xticks(range(0,20,19),np.round(np.linspace(0.01,2.0,2),1),fontsize=9)
    axs[1].set_yticks(range(0,20,19),np.round(np.linspace(0.01,2.0,2),1),fontsize=9)
    axs[1].invert_yaxis()
    fig.colorbar(pcm, ax=axs[1],shrink=0.5)

    save_fig_path=os.path.join(savetofolder,"sim4","diff_plot_k1a="+str(idx+1.0)+").png")
    fig.savefig(save_fig_path,bbox_inches='tight')

def colormap_plot_2D_fb(simulation_parameters,
                     s_perf,a_perf,b_perf,
                  last_K_trials = 15,title="",interoceptive_plot=False,
                  save_fig_path=None,name="",showtitle=False,max_state=3.5):
    
    a_perf[np.isnan(a_perf)] = 0.0

    array_of_final_states = np.mean(s_perf[:,:,:,-last_K_trials:,:],axis = (-1,-2,-3))
    array_of_final_a = np.mean(a_perf[:,:,:,-1],axis = (-1))
    array_of_final_b = np.mean(b_perf[:,:,:,-1],axis = (-1))
    max_b = np.max(b_perf[...,0])
    min_b = np.min(array_of_final_b)

    # getting the original colormap using cm.get_cmap() function 
    orig_map=plt.cm.get_cmap('viridis') 
    reversed_map = orig_map.reversed() 

    fig2,axes = plt.subplots(1,3,sharey=True,dpi=100)
    if(showtitle):
        fig2.suptitle(title,fontsize = 10)
    # fig2.tight_layout()
    # fig2.subplots_adjust(top=0.95)

    s_dist = axes[0].imshow(array_of_final_states,vmin=0,interpolation='nearest',vmax=max_state)
    axes[0].set_title("Avg. cognitive state \n (last 15 trials)",fontsize = 8)
    axes[0].set_ylabel("$\sigma_{process}$",fontsize = 8)
    axes[0].set_xlabel("$\sigma_{model}$",fontsize = 8)

    b_dist = axes[1].imshow(array_of_final_b,interpolation='nearest',cmap = reversed_map,
                                vmin=0,vmax=max_b)
    axes[1].set_title("Transition model error (after training)",fontsize = 8)
    axes[1].set_xlabel("$\sigma_{model}$",fontsize = 8)



    a_dist = axes[2].imshow(array_of_final_a,vmin=0,interpolation='nearest',cmap = reversed_map)
    if interoceptive_plot:
        axes[2].set_title("Final interoceptive model error",fontsize = 8)
    else :
        axes[2].set_title("Final feedback model error",fontsize = 8)
    axes[2].set_xlabel("$\sigma_{model}$",fontsize = 8)

    
    x_axis = simulation_parameters[:,0,0]
    y_axis = simulation_parameters[0,:,1]
    for ax in axes :
        ax.set_yticks(range(x_axis.shape[0]))
        ax.set_yticklabels(np.round(x_axis,1),fontsize = 3)

        ax.set_xticks(range(y_axis.shape[0]))
        ax.set_xticklabels(np.round(y_axis,1),fontsize = 3)
    ax.invert_yaxis()

    fig2.tight_layout()
    shrinkage_factor = 0.33
    fig2.colorbar(s_dist, ax=axes[0],shrink=shrinkage_factor)
    fig2.colorbar(b_dist, ax=axes[1],shrink=shrinkage_factor)
    fig2.colorbar(a_dist, ax=axes[2],shrink=shrinkage_factor)

    if not(save_fig_path == None):
        if not os.path.exists(save_fig_path):
            os.makedirs(save_fig_path)
        fig2.savefig(os.path.join(save_fig_path,name),bbox_inches='tight')
    return fig2

def boxplot(simulation_parameters,
            s_perf,a_perf,b_perf,
            fixed_feedback_quality_indexes = [0,2,4,6,8,19],
            last_K_trials = 15,
            save_fig_path=None,name=""):
    fig3,axs = plt.subplots(2,len(fixed_feedback_quality_indexes),figsize=(10,5),dpi=200,sharex=True)
    if (type(axs)!=np.ndarray):
        axs = np.array([axs])

    axs[0,0].set_ylabel("Mental state (end of training)")
    axs[1,0].set_ylabel("Model error (end of training)")
    for k,ax in enumerate(axs[0,:]):
        index = fixed_feedback_quality_indexes[k]
        final_s = np.mean(s_perf[index,:,:,-last_K_trials:,:],axis=(-1,-2))
        
        true_fb_std = simulation_parameters[index,0,0]
        feedback_belief_stds = simulation_parameters[index,:,1]

        ax.boxplot([final_s[k,:] for k in range(final_s.shape[0])], 
                showmeans=True, whis = 99,
                    meanprops={'marker':'o','markerfacecolor':'red'})
        ax.grid()
        ax.set_ylim([0.0,4.0])
        # for k in range(b_model_for_this_true_fb.shape[0]) : #iterate over feedback belief values
        #     feedback_belief_std = simulation_parameters[fixed_feedback_quality_index,k,1]
        #     ax.boxplot(b_model_for_this_true_fb[k,:], showmeans=True, whis = 99)
        # ax.invert_yaxis()
        ax.set_title("sigma_process = " + str(np.round(true_fb_std,2)),fontsize=8)
    
    
    for k,ax in enumerate(axs[1,:]):
        index = fixed_feedback_quality_indexes[k]
        b_model_for_this_true_fb = b_perf[index,:,:,-1]
        
        true_fb_std = simulation_parameters[index,0,0]
        feedback_belief_stds = simulation_parameters[index,:,1]

    
        ax.boxplot([b_model_for_this_true_fb[k,:] for k in range(b_model_for_this_true_fb.shape[0])], 
                showmeans=True, whis = 99,
                    meanprops={'marker':'o','markerfacecolor':'blue'})
        ax.set_xticks(range(feedback_belief_stds.shape[0]))
        ax.set_xticklabels(np.round(feedback_belief_stds,1),fontsize=5)
        ax.set_xlabel("sigma_model",fontsize= 7)
        ax.grid()   
        ax.set_ylim(bottom=0,top=np.max(b_perf[:,:,:,-1]))
        ax.invert_yaxis()

    if not(save_fig_path == None):
        if not os.path.exists(save_fig_path):
            os.makedirs(save_fig_path)
        fig3.savefig(os.path.join(save_fig_path,name))

    return fig3


if __name__=="__main__":
    A = actynf.normalize(np.random.random((10,8,2)))
    B = actynf.normalize(np.random.random((10,8,2)))
    print(dist_kl_dir(A,B))
    B[2,:,0] = 0.0
    from scipy.stats import entropy
    from scipy.spatial.distance import jensenshannon

    print(np.sum(jensenshannon(A,B,axis=0)))

    print(js_dir(A,B))