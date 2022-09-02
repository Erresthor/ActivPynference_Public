from re import S
from tabnanny import verbose
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import colors as mcolors
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from pyai.model.active_model_save_manager import ActiveSaveManager
from pyai.models_neurofeedback.article_1_simulations.bagherzadeh.bagherzadeh_alpha_assymetry import alpha_assymetry_model
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

def return_all_s_from(fullpath,Ntrials) :
    all_perfs = []
    for instance in os.listdir(fullpath):
        if (not("MODEL" in instance)) :
            instancepath = os.path.join(fullpath,instance)
            list_of_perf = []
            for t in range(Ntrials):
                container = ActiveSaveManager.open_trial_container(fullpath,int(instance),t,'f')
                # print(container.o)
                avg_obs = container.s
                # print(avg_obs/4.0)
                list_of_perf.append(avg_obs/2.0) # rating between 0 and 1
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
    
if __name__ == "__main__": 
    Ntrials = 100
    Ninst = 10
    overwrite = False 
    savepath = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","TEMPORARY_TEST_BED","bagherzadeh","model_8_imperfect_b_imperfect_a")
    modelnameR = "model_rnt"
    save_pathR = savepath
    modelR = alpha_assymetry_model(modelnameR,save_pathR,'right',perfect_a=False,perfect_b = False,prior_b_precision=5,prior_a_precision=1.5,verbose=True)
    modelR.index = 0
    modelR.initialize_n_layers(Ninst)
    modelR.run_n_trials(Ntrials,overwrite=overwrite,global_prop=None)


    modelnameL = "model_lnt"
    save_pathL = savepath
    modelL = alpha_assymetry_model(modelnameL,save_pathL,'left',perfect_a=False,perfect_b = False,prior_b_precision=5,prior_a_precision=1.5,verbose=True)
    modelL.index = 0
    modelL.initialize_n_layers(Ninst)
    modelL.run_n_trials(Ntrials,overwrite=overwrite,global_prop=None)
    # Exploitation time ! 
    # Too hard to use an avaluate container method due to the paradigm not being that easy !
    
    # All instances :
    fullR = os.path.join(save_pathR,modelnameR)
    fullL = os.path.join(save_pathL,modelnameL)
    
    plot_average_feedback(fullR,fullL,Ntrials)
    
    plot_average_states_L_R(fullR,fullL,Ntrials)
    

    # def manual_smooth(y,win=3):
    #     smoothed = np.zeros(y.shape)
    #     for k in range(y.shape[0]):
    #         cnt = 0
    #         summation = 0
    #         for i in range(k-win,k+win,1):
    #             if (i>=0):
    #                 try :
    #                     summation = summation + y[i]
    #                     cnt = cnt + 1
    #                 except :
    #                     pass
    #         smoothed[k] = summation/max(cnt,1)
    #     return smoothed

    # # Trial plot : 
    # right_typical_trial = np.average(perf_file_R,axis=1)
    # left_typical_trial = np.average(perf_file_L,axis=1)
    # trial_ts = np.arange(0,T)


    # # for i in range(Ninst):
    # #     #yhat = savgol_filter(right[i,:], 51, 3) # window size 51, polynomial order 3
    # #     plt.plot(np.arange(0,Ntrials,1),manual_smooth(right[i,:],3),color = 'blue')
    # color_right = np.array([0,0,1,0.3])
    # color_left = np.array([1,0,0,0.3])
    # color_artic = np.array([0,0,0,0.3])
    # color_rightP = np.array([0,0,1,0.6])
    # color_leftP = np.array([1,0,0,0.6])
    # stdRight = np.sqrt(np.var(right,axis=0)) # Actually std
    # stdLeft = np.sqrt(np.var(left,axis=0)) # Actually std

    # ts = np.arange(0,Ntrials,1)
    # avgRight = np.average(right,axis=0)
    # avgLeft = np.average(left,axis=0)
    # plt.fill_between(ts,avgRight-stdRight,avgRight+stdRight,color=color_right)
    # plt.fill_between(ts,avgLeft-stdLeft,avgLeft+stdLeft,color=color_left)
    # plt.fill_between(ts,arr_artic-std_artic,arr_artic+std_artic,color=color_artic)

    # for ins in range(Ninst):
    #     plt.scatter(np.arange(0,Ntrials,1),right[ins,:],marker='+',color=color_rightP,s=1)
    #     plt.scatter(np.arange(0,Ntrials,1),left[ins,:],marker='+',color=color_leftP,s=1)
    # plt.plot(ts,avgRight,color = 'blue')
    # plt.plot(ts,avgLeft,color = 'red')
    # plt.plot(ts,arr_artic,color='black')
    # plt.xlabel("Trials",fontsize = 15)
    # plt.ylabel("Average feedback level",fontsize = 15)
    # plt.grid()
    # plt.show()