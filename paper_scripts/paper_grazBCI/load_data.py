import sys,os
import scipy.io
import h5py
import mne

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

                if (stimtype=="S  2"): # Beginning of a task block
                    list_of_stimuli.append([1,stimuli_counter,stimtime_s,None])
                    if (stimuli_counter>0):
                        list_of_stimuli[-2][3] = stimtime_s
                    stimuli_counter = stimuli_counter+1
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
        scores.append([right_filtpower,left_filtpower,nf_laterality,nf_intensity,nf_intensity_smoothed])
    return subj_names,scores,timestamps

def load_bold_scores_xp1(xp1_data_path,_feedback_received = "eeg"):
    xp1_der_path = os.path.join(xp1_data_path,"derivatives")
    subj_names = os.listdir(xp1_der_path)
    full_paths = [os.path.join(xp1_der_path,subj_name) for subj_name in subj_names]
    score_dicts,timestamps = get_score_dicts(subj_names,full_paths,feedback_received = _feedback_received , score_mod="bold")
    scores = []
    for dictionary in score_dicts:
        # 1. EEG powered measured at the two areas of interest : 
        right_filtpower = dictionary["roimean_left"][:]
        left_filtpower = dictionary["roimean_right"][:]

        # 2. The various feedback modalities : 
        ["nf","nf_laterality","normnf_laterality","smoothnf_laterality"]
        nf_laterality = dictionary["nf_laterality"][:]
        nf_laterality_norm = dictionary["normnf_laterality"][:]
        nf_laterality_smoothed = dictionary["smoothnf_laterality"][:]
        nf_intensity = dictionary["nf"][:]
        # nf_intensity_smoothed = dictionary["smoothnf"][:]
        scores.append([right_filtpower,left_filtpower,nf_laterality,nf_laterality_norm,nf_laterality_smoothed,nf_intensity])
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

if __name__ == "__main__":
    os.chdir('C:\\Users\\annic\\Desktop\\Phd\code\\active_pynference_local\\paper_scripts\\paper_grazBCI')
    print("Working directory:", os.getcwd())

    subj = 0

    # A FEW CONSTANTS : 
    XP1_DATA_PATH = os.path.join("..","..","data","xp1")

    names,eeg,bold,ts = get_full_training_in_order(XP1_DATA_PATH,metric_idx = [3,4])
    print(names)

    plt.axhline(0,color="black")
    for name,eeg_data in zip(names,eeg):
        xs = np.linspace(0,eeg_data.shape[0],eeg_data.shape[0])
        plt.plot(xs,np.clip(eeg_data,-1,1),alpha=0.6,label = name)
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