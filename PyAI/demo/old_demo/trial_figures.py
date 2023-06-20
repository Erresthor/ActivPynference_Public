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


def plot_1():
    plot_modality = 0
    savepath = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","code","results","A_constant")
    
    foldername_old = "B_no_prior_A_straight"
    model1 = ActiveModel.load_model(os.path.join(savepath,foldername_old))
    
    foldername=  "B_no_prior_A_straight_03aug_without_the_errors_nomemdec"
    
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

    trial_plot_from_name(savepath,foldername,0,[0,1,2,22,23,24,25],title="Perfect feedback & perfect subject perception model, trial ")

    input()
    plt.close()

def plot_2():
    
    savepath = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","TEMPORARY_TEST_BED","one_more_trial")
    
    foldername = "learn_b_a_known_no_poub_low_conf_learn_a_high_conf"

    foldername = "my_new_model_oned_2"

    
    model2 = nf_model(foldername,savepath,prop_poubelle = 0.3,
                learn_a = True,prior_a_ratio = 3,prior_a_strength=1.5,
                learn_b= True,prior_b_ratio = 1.0,prior_b_strength=0.1,
                learn_d= True,
                mem_dec_type=MemoryDecayType.NO_MEMORY_DECAY,mem_dec_halftime=5000,
                perfect_a = False,perfect_b=False,verbose = False,SHAM="False")
    model2.input_parameters = "a weak, b flat, one d"
    model2.index = 0 # This is to track the input parameters
    # We can also add custom values for the matrices or modify the run here :
    # model.parameter = new_value
    Ninstances = 10 
    Ntrials = 150
    overwrite= False

    model2.initialize_n_layers(Ninstances)
    model2.verbose= True
    model2.run_n_trials(Ntrials,overwrite=overwrite,global_prop=None)


    #foldername = "my_new_model_oned"
    # for i in range(10):
    #     trial_plot_from_name(savepath,foldername,i,[2])
    # input()
    
    trial_plot_from_name(savepath,foldername,7,[0,1,10,20,145])
    input()
    plt.close()

if __name__=="__main__":
    plot_2()