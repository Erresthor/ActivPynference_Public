from asyncio import trsock
import sys,os
import time as t
from cmath import pi
from imageio import save
import numpy as np
from pyai.layer.layer_learn import MemoryDecayType

from pyai.base.matrix_functions import *
from pyai.base.file_toolbox import load_flexible
from pyai.base.function_toolbox import spm_wnorm
from pyai.neurofeedback_run import save_model_performance_dictionnary,load_model_performance_dictionnary,evaluate_model_figure,trial_plot_from_name
import matplotlib.pyplot as plt
from pyai.models_neurofeedback.article_1_simulations.climb_stairs_flat_priors import nf_model,evaluate_container
from pyai.model.active_model import ActiveModel
from pyai.model.active_model_container import ActiveModelSaveContainer
from pyai.base.function_toolbox import spm_wnorm

if __name__=="__main__":
    plot_modality = 0
    savepath = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","code","results","A_constant")
    
    foldername_old = "B_no_prior_A_straight"
    model1 = ActiveModel.load_model(os.path.join(savepath,foldername_old))
    
    # trial_plot_from_name(savepath,foldername_old,6,[0,10,20,30,40])

    foldername=  "B_no_prior_A_straight_03aug_without_the_errors_nomemdec"
    # trial_plot_from_name(savepath,foldername,3,[0,10,20,30,40])
    # plt.show()
    model2 = nf_model(foldername,savepath,0.3,
                learn_a= False,perfect_a=True,
                learn_b= True, perfect_b= False,prior_b_ratio=1)#,mem_dec_type=MemoryDecayType.STATIC,mem_dec_halftime=100)
    model2.C = model1.C

    model2.input_parameters = "a perfect, b flat, c nonlinear"
    model2.index = 0 # This is to track the input parameters
    # We can also add custom values for the matrices or modify the run here :
    # model.parameter = new_value
    Ninstances = 10 
    Ntrials = 150
    overwrite=False

    model2.initialize_n_layers(Ninstances)
    model2.verbose= True
    model2.run_n_trials(Ntrials,overwrite=overwrite,global_prop=None)

    # print()
    # the_b = model2.layer_list[2].b_[0]
    # novelty_b = -spm_wnorm(the_b)
    # print(np.round(novelty_b,2)) 
    # for act in range (the_b.shape[-1]):
    #     action_b = the_b[:,:,act]
    #     explored = np.sum(action_b,axis=0)
        
    #     print(np.round(explored,2))


    #evaluate_model_figure(evaluate_container,savepath,foldername)
    #trial_plot_from_name(savepath,foldername,3,[0,10,50,100],title="Perfect feedback & perfect subject perception model, trial ")
    #trial_plot_from_name(savepath,foldername,3,[0,10,20],title="Perfect feedback & perfect subject perception model, trial ")

    trial_plot_from_name(savepath,foldername,0,[0,1,2,15,20],title="Perfect feedback & perfect subject perception model, trial ")

    input()
    plt.close()
    # print(model2.a[0]-model1.a[0])

    # print("---------")
    # print(model2.A[0]-model1.A[0])

    # print("---------")
    # print(model2.b[0]-model1.b[0])

    # print("---------")
    # print(model2.B[0]-model1.B[0])

    # print("---------")
    # print(model2.C[0]-model1.C[0])

    # print("---------")
    # print(model2.D[0]-model1.D[0])

    # print(model1.layer_options.decay_half_time)
    # print(model2.layer_options.memory_decay)
