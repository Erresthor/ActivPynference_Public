from re import S
from tabnanny import verbose
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import colors as mcolors
import os, sys, inspect
import numpy as np
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from pyai.model.active_model_save_manager import ActiveSaveManager
from pyai.models_neurofeedback.article_1_simulations.bagherzadeh.orientation_vs_intensity import bagherzadeh_model

def return_all_from(fullpath,Ntrials) :
    all_perfs = []
    for instance in os.listdir(fullpath):
        if (not("MODEL" in instance)) :
            instancepath = os.path.join(fullpath,instance)
            list_of_perf = []
            for t in range(Ntrials):
                container = ActiveSaveManager.open_trial_container(fullpath,int(instance),t,'f')
                # print(container.o)
                avg_obs = container.o
                # print(avg_obs/4.0)
                list_of_perf.append(avg_obs/4.0) # rating between 0 and 1
            all_perfs.append(list_of_perf)
    return(all_perfs)


# Generate a succession of trial results for a model in a list
# Made to work as a command and simulate a single model index
# To integrate in a cluster like environment

if __name__=="__main__":
    b_precision = list(np.arange(1.0,20.0+0.5,0.5))
    b_precision.append("perfect")


    input_arguments = sys.argv
    assert len(input_arguments)>=4,"Data generator needs at least 4 arguments : savepath and an index + n_instances and n_trials"
    name_of_script = input_arguments[0]
    save_path = input_arguments[1]
    list_index = int(input_arguments[2])
    Ninstances = int(input_arguments[3])
    Ntrials = int(input_arguments[4])

    try :
        overwrite = (input_arguments[5]== 'True')
    except :
        overwrite = False

    try :
        model_name = "model_" + str(b_precision[list_index])
    except :
        print("Index beyond model list length")
        sys.exit()
    
    # Get the index-th element of the dictionnary to generate the related 
    # Get the index-th element of the dictionnary to generate the related     
    perfect_b = (b_precision[list_index]=="perfect") 

    p_i = 0.25
    p_o = 0.5

    # THIS GENERATES THE TRIALS without using run_model
    intance_b_prec = b_precision[list_index]
    modelR = bagherzadeh_model(model_name,os.path.join(save_path,"right"),p_i,p_o,
                                            neurofeedback_training_group = "right",
                                            perfect_a=False,learn_a=True,prior_a_confidence=1,prior_a_precision=1,
                                            perfect_b=perfect_b,learn_b=True,prior_b_confidence=1,prior_b_precision=intance_b_prec,
                                            learn_d=True,prior_d_precision=3,prior_d_confidence=1) 
    modelR.initialize_n_layers(Ninstances)
    
    
    modelL = bagherzadeh_model(model_name,os.path.join(save_path,"left"),p_i,p_o,
                                            neurofeedback_training_group = "left",
                                            perfect_a=False,learn_a=True,prior_a_confidence=1,prior_a_precision=1,
                                            perfect_b=perfect_b,learn_b=True,prior_b_confidence=1,prior_b_precision=intance_b_prec,
                                            learn_d=True,prior_d_precision=3,prior_d_confidence=1) 
    modelL.initialize_n_layers(Ninstances)
    
    modelR.run_n_trials(Ntrials,overwrite=overwrite,global_prop=None)
    modelL.run_n_trials(Ntrials,overwrite=overwrite,global_prop=None)