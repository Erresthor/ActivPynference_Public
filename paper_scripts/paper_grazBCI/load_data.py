import sys,os
import scipy.io
import h5py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import actynf

def compute_time_from_stamp(stamp):
    f_sample = 200.0 # Hz, indpendent of modality used
    return (stamp - 1.0) * (1/f_sample) # in seconds

def extract_markers(_vmrk_path):
    """
    Very ugly function to open a .vmrk file !
    """
    FIXED_RUN_TIME = 20.0
    
    data = pd.read_csv(_vmrk_path, header=None, skiprows=4, delimiter='\t')

    list_of_stimuli = [] # List of "Index | Training or rest | Start | End"
    stimuli_counter = 0
    for i, row in data.iterrows():
        for colname in data.columns:
            row_element = row[colname]
            if ("Stimulus" in row_element):
                list_of_info = (row_element.split(","))
                stimtype = (list_of_info[1])
                stimtime = int(list_of_info[2])
                
                stimtime_s = compute_time_from_stamp(stimtime)
                if (stimtype=="S 99"): # Beginning of a rest block
                    list_of_stimuli.append([0,stimuli_counter,stimtime_s,None])
                    if (stimuli_counter>0):
                        list_of_stimuli[-2][3] = stimtime_s
                    stimuli_counter = stimuli_counter+1

                elif (stimtype=="S  2"): # Beginning of a task block
                    
                    if stimuli_counter==0:
                        # Problem : we should start by a S99: add one 20s before !
                        # The marker file contains the list of markers assigned to the EEG recordings and 
                        # their properties (marker type, marker ID and position in data points).
                        # Three type of markers are relevant for the EEG processing: R128 (Response): 
                        # is the fMRI volume marker to correct for the gradient artifact S 99 (Stimulus): 
                        # is the protocol marker indicating the start of the Rest block S 2 (Stimulus): 
                        # is the protocol marker indicating the start of the Task (Motor Execution Motor 
                        # Imagery or Neurofeedback)
                        # Warning : in few EEG data, the first S99 marker might be missing,
                        # but can be easily “added” 20 s before the first S 2.
                        # Source : https://openneuro.org/datasets/ds002336/versions/2.0.2
                        list_of_stimuli.append([0,stimuli_counter,stimtime_s-FIXED_RUN_TIME,None])
                        stimuli_counter += 1
                    
                    list_of_stimuli.append([1,stimuli_counter,stimtime_s,None])
                    
                    if (stimuli_counter>0):
                        list_of_stimuli[-2][3] = stimtime_s
                    stimuli_counter = stimuli_counter+1
                else :
                    list_of_stimuli.append([-1,0,stimtime_s,None])
    # Finally, we assume that the last block ends exactly 20 seconds after it started :
    list_of_stimuli[-1][3] = list_of_stimuli[-1][2] + FIXED_RUN_TIME
    
    return list_of_stimuli

def get_score_dicts(subj_names,full_paths,feedback_received = "eeg",score_mod="eeg"):
    """
    Feedback reveived can be > eeg, > fmri or > eegfmri
    Score modality can be > eeg or > bold
    
    """
    # if xp_code=='xp1':
    #     full_paths = XP1_SUBJ_PATHS
    #     subj_names = XP1_SUBJ_NAMES
    # else : 
    #     raise NotImplementedError("Yet to implement XP2")
    
    all_scores = []
    timestamp_list = []
    for k,subj_name in enumerate(subj_names):
        # NF scores
        dataname = "d_" + subj_name + "_task-"+ feedback_received + "NF_NF"+score_mod +"_scores.mat"
        mat_path_test = os.path.join(full_paths[k],"NF_"+score_mod,dataname)
        all_scores.append(h5py.File(mat_path_test)["NF_"+score_mod])

        # EEG markers
        vmrk_path = os.path.join(full_paths[k],"eeg_pp",subj_name+"_task-"+ feedback_received + "NF_eeg_pp.vmrk")
        timestamp_list.append(extract_markers(vmrk_path))
    return all_scores,timestamp_list

def load_eeg_scores_xp1(xp1_data_path,_feedback_received = "eeg"):
    
    # NF_eeg 
    # → .nf_laterality (NF score computed as for real-time calculation - equation (1))
    # → .filteegpow_left (Bandpower of the filtered eeg signal in C1) 
    # → .filteegpow_right (Bandpower of the filtered eeg signal in C2) 
    # → .nf (vector of NF scores -4 per s- computed as in eq 3) for comparison with XP2 
    # → .smoothed
    # → .eegdata (64 X 200 X 400 matrix, with the pre-processed EEG signals according to the steps described above) 
    # → .method
    
    xp1_der_path = os.path.join(xp1_data_path,"derivatives")
    subj_names = os.listdir(xp1_der_path)
    full_paths = [os.path.join(xp1_der_path,subj_name) for subj_name in subj_names]
    score_dicts,timestamps = get_score_dicts(subj_names,full_paths,feedback_received = _feedback_received , score_mod="eeg")
    scores = []
    for dictionary in score_dicts:
        # 1. EEG powered measured at the two areas of interest : 
        right_filtpower = dictionary["filtpower_left"][:]
        left_filtpower = dictionary["filtpower_right"][:]

        # 2. The various feedback modalities : 
        nf_laterality = dictionary["nf_laterality"][:]
        
        nf_intensity = dictionary["nf"][:]
        nf_intensity_smoothed = dictionary["smoothnf"][:]
        
        eeg_pp = dictionary["eegdata"][:]
        eeg_pp = eeg_pp[:400,...]
        scores.append([right_filtpower,left_filtpower,nf_laterality,nf_intensity,nf_intensity_smoothed,eeg_pp])
    return subj_names,scores,timestamps

def load_bold_scores_xp1(xp1_data_path,_feedback_received = "eeg"):
    #NF_bold 
    # → .nf_laterality (calculated as for online NF calculation) 
    # → .smoothnf_laterality 
    # → .normnf_laterality 
    # → .nf (calculated as for online NF calculation in XP2) 
    # → .roimean_left (averaged BOLD signal in the left motor ROI) 
    # → .roimean_right (averaged BOLD signal in the right motor ROI) 
    # → .bgmean (averaged BOLD signal in the background slice) 
    # → .method
    
    xp1_der_path = os.path.join(xp1_data_path,"derivatives")
    subj_names = os.listdir(xp1_der_path)
    full_paths = [os.path.join(xp1_der_path,subj_name) for subj_name in subj_names]
    score_dicts,timestamps = get_score_dicts(subj_names,full_paths,feedback_received = _feedback_received , score_mod="bold")
    scores = []
    for dictionary in score_dicts:
        # 1. EEG powered measured at the two areas of interest : 
        right_filtpower = dictionary["roimean_left"][:]
        left_filtpower = dictionary["roimean_right"][:]
        bg_mean = dictionary["bgmean"][:]

        # 2. The various feedback modalities : 
        # ["nf","nf_laterality","normnf_laterality","smoothnf_laterality"]
        nf_laterality = dictionary["nf_laterality"][:]
        nf_laterality_norm = dictionary["normnf_laterality"][:]
        nf_laterality_smoothed = dictionary["smoothnf_laterality"][:]
        nf_intensity = dictionary["nf"][:]
        # nf_intensity_smoothed = dictionary["smoothnf"][:]
        scores.append([right_filtpower,left_filtpower,bg_mean,nf_laterality,nf_laterality_norm,nf_laterality_smoothed,nf_intensity])
    return subj_names,scores,timestamps

def get_full_training_in_order(xp1_data_path,metric_idx = [3,4]):
    # What recorded metric is of interest to us in [eef,bold]
    names,_,ts = load_eeg_scores_xp1(xp1_data_path,"eeg")

    LOAD_ORDER = [
        ["fmri","eeg","eegfmri"], # Subject 1
        ["eeg","eegfmri","fmri"], # Subject 2
        ["fmri","eeg","eegfmri"], # Subject 3
        ["fmri","eegfmri","eeg"], # Subject 4
        ["eegfmri","fmri","eeg"], # Subject 5
        ["eegfmri","eeg","fmri"], # Subject 6
        ["eeg","fmri","eegfmri"], # Subject 7
        ["eeg","fmri","eegfmri"], # Subject 8
        ["fmri","eegfmri","eeg"], # Subject 9
        ["eegfmri","fmri","eeg"]  # Subject 10
    ]

    eeg_recordings,bold_recordings =  [],[]
    for subj_id,name in enumerate(names):
        eeg_data = []
        bold_data = []
        for task_code in LOAD_ORDER[subj_id]:
            names,eeg_scores,_ = load_eeg_scores_xp1(xp1_data_path,task_code)
            assert name==names[subj_id], "Error in names ordering ? " + name + " =\= " + str(names[subj_id])
            eeg_data.append(eeg_scores[subj_id][metric_idx[0]])

            names,bold_scores,_ = load_bold_scores_xp1(xp1_data_path,task_code)
            assert name==names[subj_id], "Error in names ordering ? " + str(names[subj_id])
            bold_data.append(bold_scores[subj_id][metric_idx[1]])
        eeg_data = np.concatenate(eeg_data,axis=0)[:,0]
        bold_data = np.concatenate(bold_data,axis=0)[:,0]

        eeg_recordings.append(eeg_data)
        bold_recordings.append(bold_data)
    return names,eeg_recordings,bold_recordings,ts


def get_all_derivatives_in_order(xp1_data_path):
    # What recorded metric is of interest to us in [eef,bold]
    names,_,_ = load_eeg_scores_xp1(xp1_data_path,"eeg")

    LOAD_ORDER = [
        ["fmri","eeg","eegfmri"], # Subject 1
        ["eeg","eegfmri","fmri"], # Subject 2
        ["fmri","eeg","eegfmri"], # Subject 3
        ["fmri","eegfmri","eeg"], # Subject 4
        ["eegfmri","fmri","eeg"], # Subject 5
        ["eegfmri","eeg","fmri"], # Subject 6
        ["eeg","fmri","eegfmri"], # Subject 7
        ["eeg","fmri","eegfmri"], # Subject 8
        ["fmri","eegfmri","eeg"], # Subject 9
        ["eegfmri","fmri","eeg"]  # Subject 10
    ]

    eeg_recordings,bold_recordings,timestamps =  [],[],[]
    
    for subj_id,name in enumerate(names):
        
        eeg_data = []
        bold_data = []
        timestamp_subject = []
        for task_code in LOAD_ORDER[subj_id]:
            names,eeg_scores,tmstps_task = load_eeg_scores_xp1(xp1_data_path,task_code)
            assert name==names[subj_id], "Error in names ordering ? " + name + " =\= " + str(names[subj_id])
            eeg_data.append(eeg_scores[subj_id])

            names,bold_scores,_ = load_bold_scores_xp1(xp1_data_path,task_code)
            assert name==names[subj_id], "Error in names ordering ? " + str(names[subj_id])
            bold_data.append(bold_scores[subj_id])
            
            timestamp_subject.append(tmstps_task[subj_id])
        # eeg_data = np.concatenate(eeg_data,axis=0)[:,0]
        # bold_data = np.concatenate(bold_data,axis=0)[:,0]

        eeg_recordings.append(eeg_data)
        bold_recordings.append(bold_data)
        timestamps.append(timestamp_subject)
    return names,eeg_recordings,bold_recordings,timestamps




# Methods from plot_data.ipynb : 
def observation_histogram(x,bins):
    _digitized_points = np.digitize(x,bins)-1
    
    # Let's mask out the nans : 
    _digitized_points = _digitized_points[~np.isnan(x)]
    
    
    # Build a histogram from each of those digitized values : 
    # We can even play with weights to account for more complex perception rules
    # (for now, no weights)
    obs_histogram = np.bincount(_digitized_points,minlength=bins.shape[0]-1)
    
    if(np.sum(obs_histogram)==0): # No points here ! Assume that we got any possible observation but it was not seen by our subject !
        return np.ones_like(obs_histogram),0.0
    
    return obs_histogram,1.0

def discretize_input(_input_array,_feedback_bins,_n_actions_per_run):
    discretized_input,input_binary_filter = [],[]
    
    Nsubj,Ntasks,Ndata = _input_array.shape
    
    for subj in range(Nsubj):
        subject_discretized_input,subject_binary_filter = [],[]
                
        for task in range(Ntasks) :             
            input_task_data = _input_array[subj,task]
            
            # There are 20 different "trials"  in a task : 
            # (odds are rest trials, evens are effort trials)
            input_task_data_per_run = np.reshape(input_task_data,(20,-1))
            
            # We can reshape each individual trial to have the correct number of observation
            # sequences :
            avg_points_per_obs = input_task_data_per_run.shape[-1]/_n_actions_per_run
            N_points_per_obs = int(avg_points_per_obs)
            assert (N_points_per_obs - avg_points_per_obs)<1e-5,"Check the number of actions per run : it should be a multiple of the amount of feedback points each run. ({})".format(input_task_runs.shape[-1])
            assert N_points_per_obs > 0 , "Check the number of actions per run : there should be at least one feedback point per action"        
            
            input_data_per_observation = np.reshape(input_task_data,(20,-1,N_points_per_obs))
            
            
            Nruns,Nsteps,Npoints = input_data_per_observation.shape


            observation_histogram_list,observation_binary_list = [],[]
            for run in range(Nruns):
                observation_histogram_list.append([])
                observation_binary_list.append([])
                
                for step in range(Nsteps):
                    histogram_obs,binary_filter = observation_histogram(input_data_per_observation[run,step],_feedback_bins)
                    
                    observation_histogram_list[-1].append(histogram_obs)
                    observation_binary_list[-1].append(binary_filter)
                    
            subject_discretized_input.append(np.array(observation_histogram_list))
            subject_binary_filter.append(np.array(observation_binary_list))
            
        discretized_input.append(subject_discretized_input)
        input_binary_filter.append(subject_binary_filter)
    return actynf.normalize(np.array(discretized_input)),np.array(input_binary_filter)



def preprocess_data(eeg_data,bold_data,trial_time_stamps,options = {"n_actions_per_run" : 10,
                            "n_outcomes":{
                                "i_eeg" : 5,
                                "o_eeg" : 5,
                                "i_bold" : 5,
                                "o_bold" : 5
                            }
                        }):
    
    n_actions_per_run = options["n_actions_per_run"]
    
    
    timestamps_arr = np.array(trial_time_stamps)
    # Let's name the quantities we will use in this study :
    
    # feedbacks as seen by the subject : 
    BOLD_LATERALITY_FEEDBACK = np.array([[task_data[3] for task_data in s_data] for s_data in bold_data])
    BOLD_INTENSITY_LEFT_ERD = np.array([[task_data[6] for task_data in s_data] for s_data in bold_data])

    EEG_LATERALITY_FEEDBACK = np.array([[task_data[2] for task_data in s_data] for s_data in eeg_data])

    NORM_LEFT_BP = np.array([[task_data[1] for task_data in s_data] for s_data in eeg_data]) # Normalized band power of the filtered EEG signal in C1 (LEFT)
    EEG_INTENSITY_LEFT_ERD = np.array([[task_data[3] for task_data in s_data] for s_data in eeg_data]) # In XP2, the feedback was computed using only the lbp
    EEG_INTENSITY_LEFT_ERD_SMOOTHED = np.array([[task_data[4] for task_data in s_data] for s_data in eeg_data]) # In XP2, the feedback was computed using only the lbp 
    
    
    # Let's clip and discretize these observations :

    # Main parameter : how many actions per run (how many timesteps per obs ?)
    

    # There is a fixed amount of runs per task (10):

    # Each of those is made of a rest and an effort sub-run (10s).
    # During each sub run, subjects get a total of 1600/20 = 80 EEG feedback values
    # and 200/20 = 10 BOLD feedback values. 
    # Assuming that each mental action occurs regularly, we can 
    # separate each set of values into a number of individual observations :

    # One feedback bin for each observation modality : 
    # EEG INTENSITY : 
    erd_intensity_eeg = np.clip(EEG_INTENSITY_LEFT_ERD,-1.0,1.0)[...,0]
    feedback_bins = np.linspace(-1.001,1.001,options["n_outcomes"]["i_eeg"]+1)
    o_eeg_int,bin_eeg_int = discretize_input(erd_intensity_eeg,feedback_bins,n_actions_per_run)
    
    # EEG LATERALITY
    eeg_feedbacks = np.clip(EEG_LATERALITY_FEEDBACK,-1.0,1.0)[...,0]
    feedback_bins = np.linspace(-1.001,1.001,options["n_outcomes"]["o_eeg"]+1)
    o_eeg_lat,bin_eeg_lat = discretize_input(eeg_feedbacks,feedback_bins,n_actions_per_run)
    
    # BOLD INTENSITY
    erd_intensity_bold = np.clip(BOLD_INTENSITY_LEFT_ERD,-1.0,1.0)[...,0]
    feedback_bins = np.linspace(-1.001,1.001,options["n_outcomes"]["i_bold"]+1)
    o_bold_int,bin_bold_int = discretize_input(erd_intensity_bold,feedback_bins,n_actions_per_run)
    
    # BOLD LATERALITY
    bold_feedbacks = np.clip(BOLD_LATERALITY_FEEDBACK,-1.0,1.0)[...,0]
    feedback_bins = np.linspace(-1.001,1.001,options["n_outcomes"]["o_bold"]+1)
    o_bold_lat,bin_bold_lat = discretize_input(bold_feedbacks,feedback_bins,n_actions_per_run)

    
    

    timestamps_arr = np.array(trial_time_stamps)
    
    return {
        "time":timestamps_arr,
        "eeg":{
            "intensity":{
                "val":o_eeg_int,
                "filter" :bin_eeg_int
            },
            "laterality":{
                "val":o_eeg_lat,
                "filter" :bin_eeg_lat
            }
        },
        "bold":{
            "intensity":{
                "val":o_bold_int,
                "filter" :bin_bold_int
            },
            "laterality":{
                "val":o_bold_lat,
                "filter" :bin_bold_lat
            }
        }
    }
    

if __name__ == "__main__":
    os.chdir('C:\\Users\\annic\\OneDrive\\Bureau\\MainPhD\code\\ActivPynference_Public\\paper_scripts\\paper_grazBCI')
    print("Working directory:", os.getcwd())

    subj = 0

    # A FEW CONSTANTS : 
    XP1_DATA_PATH = os.path.join(".","data_downloaded")

    # names,eeg,bold,ts = get_full_training_in_order(XP1_DATA_PATH,metric_idx = [3,4])
    names,eeg,bold,ts = get_full_training_in_order(XP1_DATA_PATH,metric_idx = [0,4])
    print(names)

    plt.axhline(0,color="black")
    for name,eeg_data in zip(names,eeg):
        xs = np.linspace(0,eeg_data.shape[0],eeg_data.shape[0])
        plt.plot(xs,eeg_data,alpha=0.6,label = name)
        plt.show()
        
    # # xs = np.linspace(0,bold_data.shape[0],bold_data.shape[0])
    # # plt.plot(xs,bold_data)
    # # plt.show()


    # # fig,axes = plt.subplots(2,1,sharex=True)

    # # names,scores_eeg,ts = load_eeg_scores_xp1(XP1_DATA_PATH,"eeg")
    # # xs = np.linspace(0,ts[subj][-1][3],scores_eeg[subj][2].shape[0])
    # # axes[0].plot(xs,scores_eeg[subj][2])
    # # axes[0].legend()
    # # axes[0].grid()

    # # names,scores,ts = load_bold_scores_xp1(XP1_DATA_PATH,"eeg")
    # # xs = np.linspace(0,ts[subj][-1][3],scores[subj][2].shape[0])
    # # axes[1].plot(xs,scores[subj][4])
    # # axes[1].legend()
    # # axes[1].grid()

    # # fig.show()
    # # input()