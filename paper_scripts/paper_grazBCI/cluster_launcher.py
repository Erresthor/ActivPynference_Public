# Let's import the needed packages for further use ! 
import sys,os
import numpy as np
import copy
import matplotlib.pyplot as plt
# %matplotlib notebook

import actynf

# A few helper functions ...
from tools import clever_running_mean
from tools import save_output,extract_training_data
from tools import to_list_of_one_hots
from tools import imshow_with_labels,plot_trials
from model import gaussian_from_distance_matrix
from model import nf_net_dist as nf_phase_layer
from model import rest_phase_layer

# A FEW CONSTANTS : 
REST_CST = 0.01

def training(Ntrials,_network_rest,_network_nf):
    """
    A function to describe the training procedure : 
    """
    return_stm,return_weights = [],[]

    
    def communicate(from_net,from_stm,to_net):
            # The last state of the previous iteration becomes the first of the next
            # We also transfer the learnt matrices !
            to_net.layers[1].b = copy.deepcopy(from_net.layers[1].b)
            to_net.layers[1].d = copy.deepcopy(from_net.layers[1].d)
            to_net.layers[0].d = to_list_of_one_hots(from_stm[0].x[:,-1],to_net.layers[0].Ns)

    # We gather the initial weights of the layer at index 0
    return_weights.append(_network_nf.get_current_layers_weights())
    return_stm.append(None) # To match the indices
    for trial in range(Ntrials):
            # Phase 1 : rest !
            # Assume the agent picked actions randomly and the provide:
            rest_stm,rest_wght = _network_rest.run(False,return_STMs=True,return_weights=True)
            
            # Ensure layer continuity between phases.
            communicate(_network_rest,rest_stm,_network_nf)

            nf_stm,nf_wght = _network_nf.run(False,return_STMs=True,return_weights=True)

            # Ensure layer continuity between phases.
            communicate(_network_nf,nf_stm,_network_rest)

            return_stm.append(rest_stm)
            return_stm.append(nf_stm)

            return_weights.append(rest_wght)
            return_weights.append(nf_wght)
    
    return return_stm,return_weights

def ERDs(normed_orientation,normed_intensity,rest_cst=1.0):
        """ 
        We assume that the ERDs in the left vs right sensorimotor area are drivent by two cognitive components:
        - Intensity 
        - Orientation
        """
        PI = 3.1415
        l_erd = normed_intensity*np.cos(normed_orientation*PI/2.0) + rest_cst  # between krest and krest + 1.0
        r_erd = normed_intensity*np.sin(normed_orientation*PI/2.0) + rest_cst
        return r_erd,l_erd

def get_aai(normed_orientation,normed_intensity,krest):
        """
        A cognitive to feedback model function, based only on the laterality of the right hand motor brain activity.
        2 mental states :
        - Intensity
        - Orientation -- normed_orientation is a measure of angle between 0 and 90°   
        We have : 
        rh_mi = int * cos(ori)
        lh_mi = int * sin(ori)  
        """
        r_erd,l_erd = ERDs(normed_orientation,normed_intensity,krest)
        aai =  (l_erd - r_erd)/(l_erd + r_erd)  # Minus
        return (aai+1)/2.0  # Normalized to 0 -> 1

def distance_matrix_based_on_laterality(_Ns,krest=0.01,linear_dependence_on_intensity = False):
        _dist_mat = np.zeros(tuple(_Ns))
        for idx,val in np.ndenumerate(_dist_mat):
                intens,ori = idx[0],idx[1]

                normed_intensity = intens/(_Ns[0]-1.0)
                normed_laterality = ori/(_Ns[1]-1.0)
                        # From 0 to 1, how much orientated to the right is the mental imagery of the subject ?
                

                aai = get_aai(normed_laterality,normed_intensity,krest)
                if linear_dependence_on_intensity :
                        aai = aai * normed_intensity
                _dist_mat[idx] = 1.0 - aai
        return _dist_mat

def distance_matrix_based_on_intensity(_Ns,krest=0.01):
        _dist_mat = np.zeros(tuple(_Ns))
        for idx,val in np.ndenumerate(_dist_mat):
                intens,ori = idx[0],idx[1]
                
                intensity = intens/(_Ns[0]-1.0)
                norm_ori = ori/(_Ns[1]-1.0)
                        # From 0 to 1, how much orientated to the right is the mental imagery of the subject ?
                
                
                r_erd,l_erd = ERDs(norm_ori,intensity,krest)

                _dist_mat[idx] = 1.0 - (l_erd - krest)
                        # Left ERD  = 1.0 is good !
        return _dist_mat


def generate_a_parameter_list(b_intens,b_ori) :
    # Undordered dictionnaries are soooo not cool :(
    simulation_index_list = []    
    for b_i in range(len(b_intens)):
        for b_alpha in range(len(b_ori)):
            this_sim = {
                "filename" : "_" + str(np.round(b_intens[b_i],2)) + "_" + str(np.round(b_ori[b_alpha],2))+"_.simu",
                "b_intens" : b_intens[b_i],
                "b_ori" : b_ori[b_alpha]
            }

            simulation_index_list.append(this_sim)
    return simulation_index_list

if __name__ == "__main__":
    input_arguments = sys.argv
    assert len(input_arguments)>=3,"Data generator needs at least 3 arguments : savepath and an index + overwrite"
    name_of_script = input_arguments[0]
    save_folder = input_arguments[1]
    list_index = int(input_arguments[2])
    try :
        overwrite = (input_arguments[3]== 'True')
    except :
        overwrite = False
    
    # PARRALLELIZED SIMULATION OPTIONS
    K = 20+1
    true_fb_std_array = list(np.linspace(0.0,2.0,K))
    belief_fb_std_array = list(np.linspace(0.0,2.0,K))
    full_parameter_list = generate_a_parameter_list(true_fb_std_array,belief_fb_std_array)
    try :
        param_dict = full_parameter_list[list_index]
        b_pre_ori = param_dict["b_ori"]
        b_pre_intensity = param_dict["b_intens"]
        filename = param_dict["filename"]
        savepath = os.path.join(save_folder,filename)
    except : # Exit if sim index is beyond length
        print("[" + name_of_script + "] - Index beyond model list length")
        sys.exit()
    print(full_parameter_list)
    # savepath = os.path.join("temp","here.simu")
    # overwrite = False
    # b_pre_intensity = 1.0
    # b_pre_ori = 50.0


    Ntrials = 200
    Nsubjects = 10

    sigmas_proc = [1.5,1.5]
    observation_stickiness = 100
    p_dec = 0.1

    transition_concentration = 1.0
    transition_stickiness = 1.0


    # Model STATIC parameters
    T = 40  # The amount of timesteps for a specific trial
            # 20 s / 250 ms = 80 timesteps

    Th = 2  # Action planning temporal horizon
    Ns_proc,Ns_subj = [4,5],[4,5]
    resting_states = [0,int(Ns_proc[0]/2.0)]
    Nos = [5,5] # A single feedback with 10 distinct possible observations

    true_dist_mat = [distance_matrix_based_on_laterality(Ns_proc,linear_dependence_on_intensity = False),distance_matrix_based_on_intensity(Ns_proc)]

    sigmas_subj = [0.5] # The subject expects the feedback to be not that noisy
    belief_dist_mat = [distance_matrix_based_on_laterality(Ns_subj,linear_dependence_on_intensity = True)]
    observation_concentration,observation_stickiness = 1.0,100.0


    # Defining the mental states properties
    N_up_actions,N_down_actions,N_neutral_actions = 1,1,10
    p_decay,p_effect = 0.1,0.99

    # Hyperparameters : 
    learning_space_structure=actynf.LINEAR
    gen_temp=1.0

    network_during_nftraining = nf_phase_layer(T,Th,
            Ns_proc,Ns_subj,Nos,
            sigmas_proc,true_dist_mat,
            sigmas_subj,belief_dist_mat,observation_concentration,observation_stickiness,
            N_up_actions,N_down_actions,N_neutral_actions,
            p_dec,p_effect,
            transition_concentration,transition_stickiness,
            learning_space_structure=learning_space_structure,gen_temp=gen_temp,
            resting_state_per_factor=resting_states,
            b_pre=[b_pre_intensity,b_pre_ori])

    network_during_rest = rest_phase_layer(T,Th,
            Ns_proc,Ns_subj,Nos,
            sigmas_proc,true_dist_mat, 
            sigmas_subj,belief_dist_mat,observation_concentration,observation_stickiness,
            N_up_actions,N_down_actions,N_neutral_actions,
            p_dec,p_effect,
            transition_concentration,transition_stickiness,
            learning_space_structure=learning_space_structure,gen_temp=3.0,
            resting_state_per_factor=resting_states,
            b_pre=[b_pre_intensity,b_pre_ori])

    # This is a straight forward run ! 
    exists = os.path.isfile(savepath)
    if (not(exists)) or (overwrite):
            stm_subjs,weight_subjs = [],[]
            for sub in range(Nsubjects):
                    print("Subject " + str(sub) + " / " + str(Nsubjects))
                    subj_net = network_during_nftraining.copy_network(sub)
                    rest_net = network_during_rest.copy_network(sub)
                    STMs,weights = training(Ntrials,rest_net,subj_net)
                    stm_subjs.append(STMs)
                    weight_subjs.append(weights)
            save_output(stm_subjs,weight_subjs,savepath)

