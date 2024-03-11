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
        right_filtpower = dictionary["filtpower_left"][:]
        left_filtpower = dictionary["filtpower_right"][:]

        # 2. The various feedback modalities : 
        nf_laterality = dictionary["nf_laterality"][:]
        nf_intensity = dictionary["nf"][:]
        nf_intensity_smoothed = dictionary["smoothnf"][:]
        scores.append([right_filtpower,left_filtpower,nf_laterality,nf_intensity,nf_intensity_smoothed])
    return subj_names,scores,timestamps

if __name__ == "__main__":
    os.chdir('C:\\Users\\annic\\Desktop\\Phd\code\\active_pynference_local\\paper_scripts\\paper_grazBCI')
    print("Working directory:", os.getcwd())


    # A FEW CONSTANTS : 
    XP1_DATA_PATH = os.path.join("..","..","data","xp1")

    names,scores,ts = load_eeg_scores_xp1(XP1_DATA_PATH,"eeg")
    print(ts)