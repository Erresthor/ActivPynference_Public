from re import S
from tabnanny import verbose
from turtle import width
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import colors as mcolors
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from pyai.model.active_model_save_manager import ActiveSaveManager
from pyai.models_neurofeedback.article_1_simulations.bagherzadeh.orientation_vs_intensity import neurofeedback_model
from scipy.signal import savgol_filter

def return_perf_from (fullpath,Ntrials):
    all_perfs = []
    for instance in os.listdir(fullpath):
        if (not("MODEL" in instance)) :
            instancepath = os.path.join(fullpath,instance)
            list_of_perf = []
            for t in range(Ntrials):
                container = ActiveSaveManager.open_trial_container(fullpath,int(instance),t,'f')
                # print(container.o)
                avg_obs = np.average(container.o)
                # print(avg_obs/4.0)
                list_of_perf.append(avg_obs/4.0) # rating between 0 and 1
            all_perfs.append(list_of_perf)
    return(all_perfs)

def return_all_from(fullpath,Ntrials) :
    all_perfs = []
    for instance in os.listdir(fullpath):
        if (not("MODEL" in instance)) :
            instancepath = os.path.join(fullpath,instance)
            list_of_perf = []
            for t in range(Ntrials):
                container = ActiveSaveManager.open_trial_container(fullpath,int(instance),t,'f')
                # print(container.o)
                avg_obs = container.o
                # print(avg_obs/4.0)
                list_of_perf.append(avg_obs/4.0) # rating between 0 and 1
            all_perfs.append(list_of_perf)
    return(all_perfs)
def avg_trial_plot(save_pathR,modelnameR,save_pathL,modelnameL,Ntrials) :
    
    fileR = np.squeeze(np.array(return_all_from(os.path.join(save_pathR,modelnameR),Ntrials)))
    fileL = np.squeeze(np.array(return_all_from(os.path.join(save_pathL,modelnameL),Ntrials)))


    T = fileR.shape[2]
    ts = np.arange(0*500,T*500,500) # in ms
    instances_avg_trialsR = np.average(fileR,1)
    instances_avg_trialsL = np.average(fileL,1)

    avg_L = np.average(instances_avg_trialsL,0)
    avg_R = np.average(instances_avg_trialsR,0)
    std_L = np.sqrt(np.var(instances_avg_trialsL,0))
    std_R = np.sqrt(np.var(instances_avg_trialsR,0))
    
    color_right = np.array([0,0,1,0.3])
    color_left = np.array([1,0,0,0.3])
    color_rightP = np.array([0,0,1,0.6])
    color_leftP = np.array([1,0,0,0.6])
    plt.fill_between(ts,avg_R-std_R,avg_R+std_R,color=color_right)
    plt.fill_between(ts,avg_L-std_L,avg_L+std_L,color=color_left)
    for inst in range(Ninst):
        plt.scatter(ts,instances_avg_trialsR[inst,:],marker='+',color=color_rightP,s=3)
        plt.scatter(ts,instances_avg_trialsL[inst,:],marker='+',color=color_leftP,s=3)
    
    plt.plot(ts,avg_R,color = 'blue')
    plt.plot(ts,avg_L,color = 'red')
    plt.ylim(-0.1,1.1)
    plt.xlabel("Time (ms)",fontsize = 15)
    plt.ylabel("Average feedback level",fontsize = 15)
    plt.grid()
    plt.show()

    def manual_smooth(y,win=3):
        smoothed = np.zeros(y.shape)
        for k in range(y.shape[0]):
            cnt = 0
            summation = 0
            for i in range(k-win,k+win,1):
                if (i>=0):
                    try :
                        summation = summation + y[i]
                        cnt = cnt + 1
                    except :
                        pass
            smoothed[k] = summation/max(cnt,1)
        return smoothed

def plot_average_states_L_R(fullpathR,fullpathL,Ntrials):
    fileR = np.squeeze(np.array(return_all_s_from(os.path.join(fullpathR),Ntrials)))
    fileL = np.squeeze(np.array(return_all_s_from(os.path.join(fullpathL),Ntrials)))
    T = fileR.shape[3]
    ts = np.arange(0*500,T*500,500) # in ms


    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    instances_avg_trialsR = np.average(fileR,1)
    avg_R = np.average(instances_avg_trialsR,0)
    right_values = avg_R[0,:]
    left_values = avg_R[1,:]
    rval_std = np.std(instances_avg_trialsR,0)[0,:]
    lval_std = np.std(instances_avg_trialsR,0)[1,:]
    color_right = np.array([0,0,1,0.3])
    color_left = np.array([1,0,0,0.3])
    color_rightP = np.array([0,0,1,0.6])
    color_leftP = np.array([1,0,0,0.6])
    print(right_values.shape,rval_std.shape,ts.shape)
    ax.fill_between(ts,right_values-rval_std,right_values+rval_std,color=color_right)
    ax.fill_between(ts,left_values-lval_std,left_values+lval_std,color=color_left)
    for inst in range(Ninst):
        ax.scatter(ts,instances_avg_trialsR[inst,0,:],marker='+',color=color_rightP,s=3)
        ax.scatter(ts,instances_avg_trialsR[inst,1,:],marker='+',color=color_leftP,s=3)
    ax.plot(ts,right_values,color = 'blue')
    ax.plot(ts,left_values,color = 'red')
    ax.set_ylim(-0.1,1.1)
    ax.set_xlabel("Time (ms)",fontsize = 15)
    ax.set_ylabel("Average mental state level",fontsize = 15)
    ax.set_title("RNT")
    ax.grid()

    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    instances_avg_trialsL = np.average(fileL,1)
    avg_L = np.average(instances_avg_trialsL,0)
    right_values = avg_L[0,:]
    left_values = avg_L[1,:]
    rval_std = np.std(instances_avg_trialsL,0)[0,:]
    lval_std = np.std(instances_avg_trialsL,0)[1,:]
    color_right = np.array([0,0,1,0.3])
    color_left = np.array([1,0,0,0.3])
    color_rightP = np.array([0,0,1,0.6])
    color_leftP = np.array([1,0,0,0.6])
    print(right_values.shape,rval_std.shape,ts.shape)
    ax.fill_between(ts,right_values-rval_std,right_values+rval_std,color=color_right)
    ax.fill_between(ts,left_values-lval_std,left_values+lval_std,color=color_left)
    for inst in range(Ninst):
        ax.scatter(ts,instances_avg_trialsL[inst,0,:],marker='+',color=color_rightP,s=3)
        ax.scatter(ts,instances_avg_trialsL[inst,1,:],marker='+',color=color_leftP,s=3)
    ax.plot(ts,right_values,color = 'blue')
    ax.plot(ts,left_values,color = 'red')
    ax.set_ylim(-0.1,1.1)
    ax.set_xlabel("Time (ms)",fontsize = 15)
    ax.set_ylabel("Average mental state level",fontsize = 15)
    ax.set_title("LNT")
    ax.grid()
    plt.show()

def plot_average_feedback(fullpathR,fullpathL,Ntrials):
    right = return_perf_from(fullpathR,Ntrials)
    left = return_perf_from(fullpathL,Ntrials)
    left = np.array(left)
    right = np.array(right)
    #plot_average_feedback(right,left)

    perf_file_L = left
    perf_file_R = right
    Ninst = perf_file_L.shape[0]
    Ntrials = perf_file_L.shape[1]
    Ninst = perf_file_L.shape[0]
    Ntrials = perf_file_L.shape[1]

    def manual_smooth(y,win=3):
        smoothed = np.zeros(y.shape)
        for k in range(y.shape[0]):
            cnt = 0
            summation = 0
            for i in range(k-win,k+win,1):
                if (i>=0):
                    try :
                        summation = summation + y[i]
                        cnt = cnt + 1
                    except :
                        pass
            smoothed[k] = summation/max(cnt,1)
        return smoothed

    avg_value_article = 0.7392
    std_value_article = 0.04
    arr_artic = np.zeros(Ntrials)
    std_artic = np.zeros(Ntrials)
    for t in range(Ntrials):
        arr_artic[t] = avg_value_article
        std_artic[t] = std_value_article

    # for i in range(Ninst):
    #     #yhat = savgol_filter(right[i,:], 51, 3) # window size 51, polynomial order 3
    #     plt.plot(np.arange(0,Ntrials,1),manual_smooth(right[i,:],3),color = 'blue')
    color_right = np.array([0,0,1,0.3])
    color_left = np.array([1,0,0,0.3])
    color_artic = np.array([0,0,0,0.3])
    color_rightP = np.array([0,0,1,0.6])
    color_leftP = np.array([1,0,0,0.6])
    
    
    stdRight = np.sqrt(np.var(perf_file_R,axis=0)) # Actually std
    stdLeft = np.sqrt(np.var(perf_file_L,axis=0)) # Actually std

    ts = np.arange(0,Ntrials,1)
    avgRight = np.average(perf_file_R,axis=0)
    avgLeft = np.average(perf_file_L,axis=0)
    plt.fill_between(ts,avgRight-stdRight,avgRight+stdRight,color=color_right)
    plt.fill_between(ts,avgLeft-stdLeft,avgLeft+stdLeft,color=color_left)
    plt.fill_between(ts,arr_artic-std_artic,arr_artic+std_artic,color=color_artic)

    for ins in range(Ninst):
        plt.scatter(np.arange(0,Ntrials,1),right[ins,:],marker='+',color=color_rightP,s=1)
        plt.scatter(np.arange(0,Ntrials,1),left[ins,:],marker='+',color=color_leftP,s=1)
    plt.plot(ts,avgRight,color = 'blue')
    plt.plot(ts,avgLeft,color = 'red')
    plt.plot(ts,arr_artic,color='black')
    plt.xlabel("Trials",fontsize = 15)
    plt.ylabel("Average feedback level",fontsize = 15)
    plt.grid()
    plt.show()
    

def return_all_from(fullpath,Ntrials) :
    all_perfs = []
    all_perfo = []
    for instance in os.listdir(fullpath):
        if (not("MODEL" in instance)) :
            list_of_perfo = []
            list_of_perfs = []
            for t in range(Ntrials):
                container = ActiveSaveManager.open_trial_container(fullpath,int(instance),t,'f')
                # print(container.o)
                all_s = container.s
                all_o = container.o
                # print(avg_obs/4.0)
                list_of_perfs.append(all_s) # rating between 0 and 1
                list_of_perfo.append(all_o)
            all_perfs.append(list_of_perfs)
            all_perfo.append(list_of_perfo)
    return all_perfs,all_perfo

def plot_states_trials(s_R,s_L,T,Ntrials):
    fig,axes  = plt.subplots(2,2,sharex='col')

    across_trials = np.arange(0,Ntrials,1)
    across_timesteps = np.arange(0,T,1)

    s_R = np.array(s_R)
    s_L = np.array(s_L)
    
    color_right = np.array([0,0,1,0.3])
    color_left = np.array([1,0,0,0.3])
    color_rightP = np.array([0.5,0.5,1,0.6])
    color_leftP = np.array([1,0.5,0.5,0.6])

    orientation_accross_trials = (np.average(s_R[:,:,0,:],axis=2)/4.0)
    instance_mean_or = np.average(orientation_accross_trials,axis=0)
    instance_var = np.std(orientation_accross_trials,axis=0)
    axes[0,0].fill_between(across_trials,instance_mean_or-instance_var,instance_mean_or+instance_var,color=color_right)
    orientation_accross_trials = (np.average(s_R[:,:,1,:],axis=2)/2.0)
    instance_mean_in = np.average(orientation_accross_trials,axis=0)
    instance_var = np.std(orientation_accross_trials,axis=0)
    axes[0,1].fill_between(across_trials,instance_mean_in-instance_var,instance_mean_in+instance_var,color=color_right)
    for inst in range(Ninst):
        orientation_accross_trials = np.average(s_R[inst,:,0,:],axis=1)/4.0
        axes[0,0].scatter(across_trials,orientation_accross_trials,marker='+',color=color_rightP,s=3)
        intensity_accross_trials = np.average(s_R[inst,:,1,:],axis=1)/2.0
        axes[0,1].scatter(across_trials,intensity_accross_trials,marker='+',color=color_rightP,s=3)
    axes[0,0].plot(across_trials,instance_mean_or,color="blue")
    axes[0,1].plot(across_trials,instance_mean_in,color="blue")

    orientation_accross_trials = (np.average(s_L[:,:,0,:],axis=2)/4.0)
    instance_mean_or = np.average(orientation_accross_trials,axis=0)
    instance_var = np.std(orientation_accross_trials,axis=0)
    axes[1,0].fill_between(across_trials,instance_mean_or-instance_var,instance_mean_or+instance_var,color=color_left)
    orientation_accross_trials = (np.average(s_L[:,:,1,:],axis=2)/2.0)
    instance_mean_in = np.average(orientation_accross_trials,axis=0)
    instance_var = np.std(orientation_accross_trials,axis=0)
    axes[1,1].fill_between(across_trials,instance_mean_in-instance_var,instance_mean_in+instance_var,color=color_left)
    for inst in range(Ninst):
        orientation_accross_trials = np.average(s_L[inst,:,0,:],axis=1)/4.0
        axes[1,0].scatter(across_trials,orientation_accross_trials,marker='+',color=color_leftP,s=3)
        intensity_accross_trials = np.average(s_L[inst,:,1,:],axis=1)/2.0
        axes[1,1].scatter(across_trials,intensity_accross_trials,marker='+',color=color_leftP,s=3)
    axes[1,0].plot(across_trials,instance_mean_or,color="red")
    axes[1,1].plot(across_trials,instance_mean_in,color="red")

    
    fig.suptitle("Left vs Right attention state evolution for both groups")
    axes[0,0].set_ylabel("Spatial attention orientation",fontsize = 15)
    axes[1,0].set_ylabel("Spatial attention orientation",fontsize = 15)
    axes[0,1].set_ylabel("Attention intensity",fontsize = 15)
    axes[1,1].set_ylabel("Attention intensity",fontsize = 15)
    axes[1,0].set_xlabel("Trials",fontsize = 15)
    axes[1,1].set_xlabel("Trials",fontsize = 15)
    
    # axes[0,1].set_ylim(-0.1,1.1)
    # axes[1,1].set_ylim(-0.1,1.1)
    # axes[1,0].set_ylim(-0.1,1.1)
    # axes[0,0].set_ylim(-0.1,1.1)

    axes[0,0].set_yticks([0,0.5,1], ['LEFT', 'MIDDLE', 'RIGHT'])  # Set text labels.
    axes[1,0].set_yticks([0,0.5,1], ['LEFT', 'MIDDLE', 'RIGHT'])  # Set text labels.
    axes[0,1].set_yticks([0,0.5,1], ['LOW', 'MIDDLE', 'HIGH'])  # Set text labels.
    axes[1,1].set_yticks([0,0.5,1], ['LOW', 'MIDDLE', 'HIGH'])  # Set text labels.

    axes[0,0].grid()
    axes[1,0].grid()
    axes[0,1].grid()
    axes[1,1].grid()
    #plt.show()


def plot_states_timesteps(s_R,s_L,T,Ntrials):
    fig,axes  = plt.subplots(2,2,sharex='col')

    across_trials = np.arange(0,Ntrials,1)
    across_timesteps = np.arange(0,T,1)
    across_timesteps = across_timesteps*500

    s_R = np.array(s_R)
    s_L = np.array(s_L)
    
    color_right = np.array([0,0,1,0.3])
    color_left = np.array([1,0,0,0.3])
    color_rightP = np.array([0.5,0.5,1,0.6])
    color_leftP = np.array([1,0.5,0.5,0.6])

    orientation_accross_timesteps = (np.average(s_R[:,:,0,:],axis=1)/4.0)
    instance_mean_or = np.average(orientation_accross_timesteps,axis=0)
    instance_var = np.std(orientation_accross_timesteps,axis=0)
    axes[0,0].fill_between(across_timesteps,instance_mean_or-instance_var,instance_mean_or+instance_var,color=color_right)

    attention_accross_timesteps = (np.average(s_R[:,:,1,:],axis=1)/2.0)
    instance_mean_in = np.average(attention_accross_timesteps,axis=0)
    instance_var = np.std(attention_accross_timesteps,axis=0)
    axes[0,1].fill_between(across_timesteps,instance_mean_in-instance_var,instance_mean_in+instance_var,color=color_right)

    for inst in range(Ninst):
        orientation_accross_timesteps = np.average(s_R[inst,:,0,:],axis=0)/4.0
        axes[0,0].scatter(across_timesteps,orientation_accross_timesteps,marker='+',color=color_rightP,s=3)
        intensity_accross_timesteps = np.average(s_R[inst,:,1,:],axis=0)/2.0
        axes[0,1].scatter(across_timesteps,intensity_accross_timesteps,marker='+',color=color_rightP,s=3)
    axes[0,0].plot(across_timesteps,instance_mean_or,color="blue")
    axes[0,1].plot(across_timesteps,instance_mean_in,color="blue")

    orientation_accross_timesteps = (np.average(s_L[:,:,0,:],axis=1)/4.0)
    instance_mean_or = np.average(orientation_accross_timesteps,axis=0)
    instance_var = np.std(orientation_accross_timesteps,axis=0)
    axes[1,0].fill_between(across_timesteps,instance_mean_or-instance_var,instance_mean_or+instance_var,color=color_left)
    orientation_accross_timesteps = (np.average(s_L[:,:,1,:],axis=1)/2.0)
    instance_mean_in = np.average(orientation_accross_timesteps,axis=0)
    instance_var = np.std(orientation_accross_timesteps,axis=0)
    axes[1,1].fill_between(across_timesteps,instance_mean_in-instance_var,instance_mean_in+instance_var,color=color_left)
    for inst in range(Ninst):
        orientation_accross_timesteps = np.average(s_L[inst,:,0,:],axis=0)/4.0
        axes[1,0].scatter(across_timesteps,orientation_accross_timesteps,marker='+',color=color_leftP,s=3)
        intensity_accross_timesteps = np.average(s_L[inst,:,1,:],axis=0)/2.0
        axes[1,1].scatter(across_timesteps,intensity_accross_timesteps,marker='+',color=color_leftP,s=3)
    axes[1,0].plot(across_timesteps,instance_mean_or,color="red")
    axes[1,1].plot(across_timesteps,instance_mean_in,color="red")

    
    fig.suptitle("Left vs Right attention state mean trial for both groups")
    axes[0,0].set_ylabel("Spatial attention orientation",fontsize = 15)
    axes[1,0].set_ylabel("Spatial attention orientation",fontsize = 15)
    axes[0,1].set_ylabel("Attention intensity",fontsize = 15)
    axes[1,1].set_ylabel("Attention intensity",fontsize = 15)
    axes[1,0].set_xlabel("Time (ms)",fontsize = 15)
    axes[1,1].set_xlabel("Time (ms)",fontsize = 15)
    
    # axes[0,1].set_ylim(-0.1,1.1)
    # axes[1,1].set_ylim(-0.1,1.1)
    # axes[1,0].set_ylim(-0.1,1.1)
    # axes[0,0].set_ylim(-0.1,1.1)

    axes[0,0].set_yticks([0,0.5,1], ['LEFT', 'MIDDLE', 'RIGHT'])  # Set text labels.
    axes[1,0].set_yticks([0,0.5,1], ['LEFT', 'MIDDLE', 'RIGHT'])  # Set text labels.
    axes[0,1].set_yticks([0,0.5,1], ['LOW', 'MIDDLE', 'HIGH'])  # Set text labels.
    axes[1,1].set_yticks([0,0.5,1], ['LOW', 'MIDDLE', 'HIGH'])  # Set text labels.

    axes[0,0].grid()
    axes[1,0].grid()
    axes[0,1].grid()
    axes[1,1].grid()
    #plt.show()

if __name__ == "__main__":
    T = 10
    Ntrials = 100
    Ninst = 10
    overwrite = False 
    index = "001"
    savepath = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","TEMPORARY_TEST_BED","ori_int","bagherzadeh","model_8_imperfect_b_imperfect_a")
    
    p_i = 0.25 # probability of lowering the attentional level in case of monitoring attention
    p_o = 0.5 # probability of centering the directional attention in case of monitoring intensity
    feedback_depends_on_attention = True

    modelnameR = "model_rnt"
    save_pathR = savepath
    modelR = neurofeedback_model(modelnameR,save_pathR,p_i,p_o,'right',perfect_a=False,perfect_b = True,perfect_d=True,prior_b_precision=5,prior_a_precision=1,prior_a_confidence=0.5,verbose=True)
    modelR.index = [p_i,p_o,index]
    modelR.T = T
    modelR.initialize_n_layers(Ninst)

    modelnameL = "model_lnt"
    save_pathL = savepath
    modelL = neurofeedback_model(modelnameL,save_pathL,p_i,p_o,'left' ,perfect_a=False,perfect_b = True,perfect_d=True,prior_b_precision=5,prior_a_precision=1,prior_a_confidence=0.5,verbose=True)
    modelL.index = [p_i,p_o,index]
    modelL.T = T
    modelL.initialize_n_layers(Ninst)
    
    
    # modelR.run_n_trials(Ntrials,overwrite=overwrite,global_prop=None)
    # modelL.run_n_trials(Ntrials,overwrite=overwrite,global_prop=None)
    fullpathR = os.path.join(save_pathR,modelnameR)
    fullpathL = os.path.join(save_pathR,modelnameL)

    s_R, o_R = return_all_from(fullpathR,Ntrials)
    s_L, o_L = return_all_from(fullpathL,Ntrials)
    



    from mpl_toolkits.axisartist.axislines import SubplotZero

    ts = np.arange(0,T,1)
    fig =plt.figure()
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)
    ax.set_xlim(-1,1)
    ax.set_ylim(0,T)
    for i in range(Ninst):
        avg_states_R = (np.average(np.array(s_R[i]),0)[0,:]-2)/2
        avg_states_L = (np.average(np.array(s_L[i]),0)[0,:]-2)/2
        print(avg_states_R)
        blue_cust = np.array([0,0,1,0.4])
        red_cust = np.array([1,0,0,0.4])
        ax.plot(avg_states_R,ts,color=blue_cust,linewidth=0.5)
        ax.plot(avg_states_L,ts,color=red_cust,linewidth=0.5)
    
    avg_states_R = (np.average(np.array(s_R),(0,1))[0,:]-2)/2
    avg_states_L = (np.average(np.array(s_L),(0,1))[0,:]-2)/2
    ax.plot(avg_states_R,ts,color='blue',linewidth=2.5)
    ax.plot(avg_states_L,ts,color='red',linewidth=2.5)
    for direction in ["xzero", "yzero"]:
        # adds arrows at the ends of each axis
        ax.axis[direction].set_axisline_style("-|>")

        # adds X and Y-axis from the origin
        ax.axis[direction].set_visible(True)

    for direction in ["left", "right", "bottom", "top"]:
        # hides borders
        ax.axis[direction].set_visible(False)

    # print(len(s_R))
    # print(len(s_R[0]))
    # print(s_R[0][0].shape)
    ax.set_xlabel("Orientation")
    ax.set_ylabel("Timesteps",loc='top')
    # plt.show()
    
    

    fig =plt.figure()
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)
    ax.set_xlim(-1,1)
    ax.set_ylim(0,T)
    

    first_quarter = int(Ntrials/4.0)
    last_quarter = int(3.0*Ntrials/4.0)
    print(last_quarter)
    print(np.array(s_L).shape)
    print(np.array(s_R).shape)
    avg_states_R_1 = (np.average(np.array(s_R)[:,first_quarter:,:,:],(0,1))[0,:]-2)/2
    avg_states_R_2 = (np.average(np.array(s_R)[:,last_quarter:,:,:],(0,1))[0,:]-2)/2
    
    avg_states_L_1 = (np.average(np.array(s_L)[:,first_quarter:,:,:],(0,1))[0,:]-2)/2
    avg_states_L_2 = (np.average(np.array(s_L)[:,last_quarter:,:,:],(0,1))[0,:]-2)/2

    print(avg_states_L)
    print(avg_states_R)

    ax.plot(avg_states_R_1,ts,color='blue',linewidth=2.5,linestyle="--")
    ax.plot(avg_states_R_2,ts,color='blue',linewidth=2.5)

    ax.plot(avg_states_L_1,ts,color='red',linewidth=2.5,linestyle="--")
    ax.plot(avg_states_L_2,ts,color='red',linewidth=2.5)

    for direction in ["xzero", "yzero"]:
        # adds arrows at the ends of each axis
        ax.axis[direction].set_axisline_style("-|>")

        # adds X and Y-axis from the origin
        ax.axis[direction].set_visible(True)

    for direction in ["left", "right", "bottom", "top"]:
        # hides borders
        ax.axis[direction].set_visible(False)

    # print(len(s_R))
    # print(len(s_R[0]))
    # print(s_R[0][0].shape)
    ax.set_xlabel("Orientation")
    ax.set_ylabel("Timesteps",loc='top')
    plt.show()
    

    # plot_states_trials(s_R,s_L,T,Ntrials)
    # plot_states_timesteps(s_R,s_L,T,Ntrials)
    # plt.show()
    # fig,axes  = plt.subplots(2,2,sharex='col')

    # across_trials = np.arange(0,Ntrials,1)
    # across_timesteps = np.arange(0,T,1)

    # s_R = np.array(s_R)



    # color_right = np.array([0,0,1,0.3])
    # color_left = np.array([1,0,0,0.3])
    # color_rightP = np.array([0.5,0.5,1,0.6])
    # color_leftP = np.array([1,0.5,0.5,0.6])

    # orientation_accross_trials = (np.average(s_R[:,:,0,:],axis=2)/4.0)
    # instance_mean_or = np.average(orientation_accross_trials,axis=0)
    # instance_var = np.std(orientation_accross_trials,axis=0)
    # axes[0,0].fill_between(across_trials,instance_mean_or-instance_var,instance_mean_or+instance_var,color=color_right)
    # orientation_accross_trials = (np.average(s_R[:,:,1,:],axis=2)/2.0)
    # instance_mean_in = np.average(orientation_accross_trials,axis=0)
    # instance_var = np.std(orientation_accross_trials,axis=0)
    # axes[0,1].fill_between(across_trials,instance_mean_in-instance_var,instance_mean_in+instance_var,color=color_right)
    # for inst in range(Ninst):
    #     orientation_accross_trials = np.average(s_R[inst,:,0,:],axis=1)/4.0
    #     axes[0,0].scatter(across_trials,orientation_accross_trials,marker='+',color=color_rightP,s=3)
    #     intensity_accross_trials = np.average(s_R[inst,:,1,:],axis=1)/2.0
    #     axes[0,1].scatter(across_trials,intensity_accross_trials,marker='+',color=color_rightP,s=3)
    # axes[0,0].plot(across_trials,instance_mean_or,color="blue")
    # axes[0,1].plot(across_trials,instance_mean_in,color="blue")

    # s_L = np.array(s_L)
    # orientation_accross_trials = (np.average(s_L[:,:,0,:],axis=2)/4.0)
    # instance_mean_or = np.average(orientation_accross_trials,axis=0)
    # instance_var = np.std(orientation_accross_trials,axis=0)
    # axes[1,0].fill_between(across_trials,instance_mean_or-instance_var,instance_mean_or+instance_var,color=color_left)
    # orientation_accross_trials = (np.average(s_L[:,:,1,:],axis=2)/2.0)
    # instance_mean_in = np.average(orientation_accross_trials,axis=0)
    # instance_var = np.std(orientation_accross_trials,axis=0)
    # axes[1,1].fill_between(across_trials,instance_mean_in-instance_var,instance_mean_in+instance_var,color=color_left)
    # for inst in range(Ninst):
    #     orientation_accross_trials = np.average(s_L[inst,:,0,:],axis=1)/4.0
    #     axes[1,0].scatter(across_trials,orientation_accross_trials,marker='+',color=color_leftP,s=3)
    #     intensity_accross_trials = np.average(s_L[inst,:,1,:],axis=1)/2.0
    #     axes[1,1].scatter(across_trials,intensity_accross_trials,marker='+',color=color_leftP,s=3)
    # axes[1,0].plot(across_trials,instance_mean_or,color="red")
    # axes[1,1].plot(across_trials,instance_mean_in,color="red")

    
    # fig.suptitle("Left vs Right attention state evolution for both groups")
    # axes[0,0].set_ylabel("Spatial attention orientation",fontsize = 15)
    # axes[1,0].set_ylabel("Spatial attention orientation",fontsize = 15)
    # axes[0,1].set_ylabel("Attention intensity",fontsize = 15)
    # axes[1,1].set_ylabel("Attention intensity",fontsize = 15)
    # axes[1,0].set_xlabel("Trials",fontsize = 15)
    # axes[1,1].set_xlabel("Trials",fontsize = 15)
    
    # # axes[0,1].set_ylim(-0.1,1.1)
    # # axes[1,1].set_ylim(-0.1,1.1)
    # # axes[1,0].set_ylim(-0.1,1.1)
    # # axes[0,0].set_ylim(-0.1,1.1)

    # axes[0,0].set_yticks([0,0.5,1], ['LEFT', 'MIDDLE', 'RIGHT'])  # Set text labels.
    # axes[1,0].set_yticks([0,0.5,1], ['LEFT', 'MIDDLE', 'RIGHT'])  # Set text labels.
    # axes[0,1].set_yticks([0,0.5,1], ['LOW', 'MIDDLE', 'HIGH'])  # Set text labels.
    # axes[1,1].set_yticks([0,0.5,1], ['LOW', 'MIDDLE', 'HIGH'])  # Set text labels.

    # axes[0,0].grid()
    # axes[1,0].grid()
    # axes[0,1].grid()
    # axes[1,1].grid()
    # plt.show()

