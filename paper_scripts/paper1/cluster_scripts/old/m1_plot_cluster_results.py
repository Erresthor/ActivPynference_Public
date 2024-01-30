import numpy as np
import statistics as stat
import scipy.stats as scistats
import math,sys,os,inspect
import shutil

import pickle 
import matplotlib.pyplot as plt

import actynf

from tools import clever_running_mean,color_spectrum

#!/usr/bin/python
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# Generate a succession of trial results for a model in a list
# Made to work as a command and simulate a single model index
# To integrate in a cluster like environment

def generate_a_parameter_list(A_priors,a_priors,a_weights) :
    # Undordered dictionnaries are soooo not cool :(
    simulation_index_list = []    
    for kA in range(len(A_priors)):
        for ka in range(len(a_priors)):
            for wa in range(len(a_weights)) :
                this_sim = {
                    "filename" : "_" + str(A_priors[kA]) + "_" + str(a_priors[ka])+"_"+str(a_weights[wa])+"_.simu",
                    "A_std" : A_priors[kA],
                    "a_std" : a_priors[ka],
                    "a_wgt" : a_weights[wa]
                }

                simulation_index_list.append(this_sim)
    return simulation_index_list

def extract_training_data(savepath):
    # EXTRACT TRAINING CURVES    
    with open(savepath, 'rb') as handle:
        saved_data = pickle.load(handle)
    stms = saved_data["stms"]
    weights = saved_data["matrices"]

    Nsubj = len(stms)
    Ntrials = len(weights[0])-1 # One off because we save the initial weights (= trial 0)
    return stms,weights,Nsubj,Ntrials

def initialize_list_of_list(Ns,Ny):
    z = []
    for x in range(Ns):
        z.append([])
        for y in range(Ny):
            z[-1].append(None)
    return z

def initialize_list_of_list(Ns,Ny,Nz):
    L = []
    for x in range(Ns):
        L.append([])
        for y in range(Ny):
            L[-1].append([])
            for z in range(Nz):
                L[-1][-1].append([])
    return L

def save_object_to(obj,savepath,override=True):
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))

    exists = os.path.isfile(savepath)
    if (not(exists)) or (override):
        print("Saving to " + savepath)
            
        with open(savepath, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved to :   " + savepath)

def refactor():
    """
    Change the format of the save
    """
    save_folder_name = os.path.join(os.path.abspath(os.sep),"mnt","data","Come_A","results","new_sims","003")
    all_files = [os.path.join(save_folder_name, f) for f in os.listdir(save_folder_name) if os.path.isfile(os.path.join(save_folder_name, f))]
    
    K = 20
    Kw = 10
    
    lis = initialize_list_of_list(K,K,Kw)
    extract_this = initialize_list_of_list(K,K,Kw)

    xs = []
    ys = []

    true_fb_std_array = list(np.linspace(0.001,3.0,K))
    belief_fb_std_array = list(np.linspace(0.001,3.0,K))
    belief_fb_weights_array = list(np.linspace(1.0,10.0,Kw))
    
    save_folder_name_arrayed = os.path.join(save_folder_name,"compressed")
    keys = {
        "0":true_fb_std_array,
        "1":belief_fb_std_array,
        "2":belief_fb_weights_array
    }
    save_object_to(keys,os.path.join(save_folder_name_arrayed,"KEYS"))

    for file in all_files:
        filename = os.path.basename(file)
        param_val = filename.split("_")
        print(param_val)
        
        value_1 = float(param_val[-4])  # True feedback noise
        value_2 = float(param_val[-3])  # Belief feedback noise
        value_3 = float(param_val[-2])  # Belief feedback initial weights
        
        kA = (int((value_1-0.001)/0.15))
        ka = (int((value_2-0.001)/0.15))
        kW = (int(value_3)-1)
        
        param_val[-4] = str(int(kA))
        param_val[-3] = str(int(ka))
        param_val[-2] = str(int(kW))

        new_fullname = os.path.join(save_folder_name_arrayed,(("_").join(param_val)[1:]))

        shutil.copy(file,new_fullname)
    

if __name__ == "__main__":
    savepath_a = os.path.join("simulation_outputs","cluster","RESULTS_SIM_3")
    savepath_b = os.path.join("simulation_outputs","cluster","RESULTS_SIM_5")
    savepath_arr = os.path.join("simulation_outputs","cluster","RESULTS_ARR")
    savepath_labels = os.path.join("simulation_outputs","cluster","RESULTS_LABELS")
    if(not(os.path.isfile(savepath_arr)) or not(os.path.isfile(savepath_labels))):              
        with open(savepath_a, 'rb') as handle:
            saved_object_learna = pickle.load(handle)
        params_a = saved_object_learna[0]
        results_a = np.array(saved_object_learna[1])
        
        with open(savepath_b, 'rb') as handle:
            saved_object_nolearna = pickle.load(handle)
        params_b = saved_object_nolearna[0]
        results_b = np.expand_dims(np.array(saved_object_nolearna[1]),2)

        results_c = np.concatenate([results_a,results_b],axis=2)

        params_a[2] = np.concatenate([params_a[2],np.array([100.0])],axis=-1)
        print(params_a)

        results = results_c
        save_object_to(results,savepath_arr)

        params = params_a
        save_object_to(params,savepath_labels)

    with open(savepath_arr, 'rb') as handle:
        arr = pickle.load(handle)
    with open(savepath_labels,'rb') as handle:
        labels = pickle.load(handle)
    trials_last_N = 15
    print(arr.shape)
    print(labels)

    last_N_trials = 10

    plot_those = [0,1,2,4,9,10]

    fig,axes = plt.subplots(1,len(plot_those))
    for k,plot_idx in enumerate(plot_those):
        ax = axes[k]

        # print(arr[:,:,k,0,0])
        print(arr.shape)
        mean_trial_perf = np.mean(arr[:,:,plot_idx,-last_N_trials:,:],axis=(2,3,4))
        ax.imshow(mean_trial_perf,vmin=0,vmax=3)

        ax.set_ylabel("TRUE FB std")
        ax.set_xlabel("EXPECTED FB std")
        ax.invert_yaxis()


        ax.set_yticks(range(labels[0].shape[0]))
        ax.set_yticklabels(np.round(labels[0],1),fontsize=4)

        ax.set_xticks(range(labels[1].shape[0]))
        ax.set_xticklabels(np.round(labels[1],1),fontsize=4)

        ax.set_title(labels[2][plot_idx])
    fig.show()
    input()

    # true_fb_stds = np.array(key_dict["0"])
    # belief_fb_stds = np.array(key_dict["1"])
    # belief_fb_weights = np.array(key_dict["2"])
    
    # results = initialize_list_of_list(true_fb_stds.shape[0],belief_fb_stds.shape[0],belief_fb_weights.shape[0])
    # for file in all_files:
    #     filename = os.path.basename(file)
    #     print(filename)
    #     coords = [int(k) for k in filename.split("_")[:-1]]
    #     kA = coords[0]
    #     ka = coords[1]
    #     wa = coords[2]
        
    #     _stm,_weight,_Nsubj,_Ntrials = extract_training_data(file)
    #     state_perfs = [[_stm[subj][idx][0].x[0] for idx in range(1,_Ntrials+1)] for subj in range(_Nsubj)]
    #     results[kA][ka][wa] = state_perfs
    # save_this = [[true_fb_stds,belief_fb_stds,belief_fb_weights],results]
    # save_object_to(save_this,extract_to_path)
        
    