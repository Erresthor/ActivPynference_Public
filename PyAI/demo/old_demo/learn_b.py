from pickle import FALSE
import numpy as np
import os,sys
import matplotlib.pyplot as plt
from pyai.model.model_visualizer import generate_model_sumup,general_performance_plot

from pyai.base.file_toolbox import save_flexible,load_flexible

from pyai.layer.layer_learn import MemoryDecayType

from pyai.model.active_model import ActiveModel
from pyai.model.active_model_save_manager import ActiveSaveManager
from pyai.model.active_model_container import ActiveModelSaveContainer

from pyai.neurofeedback_run import run_models,run_model,evaluate_model,evaluate_model_mean,trial_plot_from_name,evaluate_model_dict,generate_instances_figures

#from pyai.models_neurofeedback.climb_stairs import nf_model,evaluate_container

from pyai.models_neurofeedback.article_1_simulations.climb_stairs_001_2 import nf_model,evaluate_container



if __name__=="__main__":
    save_path = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","TEMPORARY_TEST_BED")
    Ninstances = 20
    Ntrials = 500
    overwrite = False
    model_name = "learn_b_a_known"
    
    # THIS GENERATES THE TRIALS without using run_model
    model = nf_model(model_name,save_path,0.3,False)
    
    model.index = -1 # This is to track the input parameters
    # We can also add custom values for the matrices or modify the run here :
    # model.parameter = new_value
    model.initialize_n_layers(Ninstances)
    trial_times = [0.01]
    model.run_n_trials(Ntrials,overwrite=overwrite,global_prop=None,list_of_last_n_trial_times=trial_times)

    # Old version : (could not shunt the way model options impacted the model :/ too restrictive or a better way of following things up ?)
    #run_model(save_path,model_name,model_options,Ntrials,Ninstances,overwrite = False,global_prop=[0,1],verbose=False)

    # Here, we shoud generate a first set of run-wide performance results
    # And save them in a dedicated file (_MODEL and _PERFORMANCES or smthg like that ?)
    #sumup = save_model_sumup_for(model_name,save_path,evaluate_container)
    sumup_dict = evaluate_model_dict(evaluator=evaluate_container,modelname=model_name,savepath=save_path)

    local_model_savepath_mean = os.path.join(save_path,model_name,"_PERFORMANCES_MEAN")
    save_flexible(sumup_dict['mean'],local_model_savepath_mean)
    local_model_savepath_var = os.path.join(save_path,model_name,"_PERFORMANCES_VAR")
    save_flexible(sumup_dict['variance'],local_model_savepath_var)
    print("Saving model results at :   " + local_model_savepath_mean+" and "+local_model_savepath_var)

    #generate_instances_figures(evaluate_container,save_path,model_name,[0,1,2,3,4,5,6,7,8,9,10])
    print(sumup_dict['mean']['observation_error'])
    print(sumup_dict['mean']['b_error'])