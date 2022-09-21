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

    trial_plot_from_name(savepath,foldername,0,[0,1,2,15,20],title="Perfect feedback & perfect subject perception model, trial ")

    input()
    plt.close()