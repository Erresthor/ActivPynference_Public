import sys,os

from cmath import pi
from imageio import save
import numpy as np
from pyai.base.matrix_functions import *
from pyai.base.function_toolbox import spm_wnorm
from pyai.neurofeedback_run import save_model_performance_dictionnary,load_model_performance_dictionnary
import matplotlib.pyplot as plt
from pyai.models_neurofeedback.article_1_simulations.climb_stairs_flat_priors import nf_model,evaluate_container

def sliding_window_mean(array_input,window_size = 5):
        list_output = np.zeros(array_input.shape)
        N = array_input.shape[0]
        for trial in range(N):
            mean_value = 0
            counter = 0
            for k in range(trial - window_size,trial + window_size + 1):
                if(k>=0):
                    try :
                        mean_value += array_input[k]
                        counter += 1
                    except :
                        a = 0
                        #Nothing lol
            list_output[trial] = mean_value/counter
        return list_output

if __name__=="__main__":
    save_path = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","TEMPORARY_TEST_BED","flat_prior_sham_feedback")
    overwrite = False
    model_name = "a_ac1.4_str1_b_ac2.8_str1"

    complete_data = True
    var = True 

    save_model_performance_dictionnary(save_path,model_name,evaluate_container,overwrite=overwrite,include_var=var,include_complete=complete_data)

    plot_modality = 0

    
    full_dico = load_model_performance_dictionnary(save_path,model_name,var,complete_data)

    field_wanted = "state_error"
    
    all_o_err = full_dico['complete'][field_wanted]
    o_err_std = np.sqrt(full_dico['variance'][field_wanted])
    o_err_mean = (full_dico['mean'][field_wanted])
    ts = np.arange(0,o_err_std.shape[0],1) 


    slided_std = sliding_window_mean(o_err_std[:,plot_modality],5)
    slided_mea = sliding_window_mean(o_err_mean[:,plot_modality],5)
    plt.fill_between(ts, slided_mea-slided_std, slided_mea+slided_std,color=np.array([0.8,0.8,1.0]))
    
    pointcolor = np.array([150.0/255,150.0/255,1.0])
    for model_indice in range(all_o_err.shape[0]):
        plt.scatter(ts,all_o_err[model_indice,:,plot_modality],c=pointcolor,s=0.33)

    # print(slided_mea,slided_std.shape)
    plt.plot(ts,slided_mea,color='red')
    #plt.xlim(0,1)
    plt.ylim(-0.1,1.1)
    plt.ylabel(field_wanted)
    plt.xlabel("Trials")
    plt.title("Evolution of " + field_wanted + " accross trials for 20 subjects with no initial b_prior and absolute a confidence")
    plt.show()
    #print(full_dico['complete']['A_list'][0])