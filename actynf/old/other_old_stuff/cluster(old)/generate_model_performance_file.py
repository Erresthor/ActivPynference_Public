from json import load
import sys,inspect,os
from pickle import FALSE
from threading import local
import numpy as np
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from pyai.base.file_toolbox import save_flexible,load_flexible
from pyai.layer.layer_learn import MemoryDecayType
from pyai.models_neurofeedback.climb_stairs import nf_model,evaluate_container
from pyai.model.active_model import ActiveModel
from pyai.neurofeedback_run import save_model_performance_dictionnary_byindex

# This file allows cluster parrallelization of performance file generation
# Normally not needed, it may prove faster than re-running the whole generation if we only want to 
# change the evaluation method !

if __name__=="__main__" :
    input_arguments = sys.argv
    assert len(input_arguments)>=2,"Data generator needs at least 2 argument : savepath, contents"
    name_of_script = input_arguments[0]
    save_path = input_arguments[1] # The folder with all the models stocked

    model_index = input_arguments[2]

    simulation_code = input_arguments[3] # Which files stocks the evaluator function needed :
    if (simulation_code=='001'):
        from pyai.models_neurofeedback.article_1_simulations.climb_stairs_flat_priors import evaluate_container as ec
    elif (simulation_code=='001_sham'):
        from pyai.models_neurofeedback.article_1_simulations.climb_stairs_flat_priors import evaluate_container as ec
    elif (simulation_code=='002'):
        from pyai.models_neurofeedback.article_1_simulations.climb_stairs_b_flat_trueA_gaussian import evaluate_container as ec
    elif (simulation_code=='003'):
        from pyai.models_neurofeedback.article_1_simulations.climb_stairs_b_flat_a_gaussian import evaluate_container as ec
    elif (simulation_code=='003_sham'):
        from pyai.models_neurofeedback.article_1_simulations.climb_stairs_b_flat_a_gaussian import evaluate_container as ec
    else :
        print("Code not recognized ... Aborting generating performance file")



    save_indicators = input_arguments[4] # 'm",'mc','mv','mcv'
    assert ('m' in  save_indicators),"The mean result should always be part of a perf file" # - Why is it an option then ? --'
                                                                                            # - Because >:(
    complete = ('c' in save_indicators)
    variance = ('v' in save_indicators)

    try : 
        overwrite = input_arguments[5] # Weither to replace existing performance files
        overwrite = (overwrite=="True")
    except :
        overwrite = False
    

    print("------------------------------------------------------------------")
    print("Generating sumup for " + save_path + " - ( model " + str(model_index) + " )")
    if(overwrite):
        print("(Overwriting previous files)")
    print("------------------------------------------------------------------")
    save_model_performance_dictionnary_byindex(save_path,model_index,ec,overwrite=overwrite,include_var=variance,include_complete=complete)
    print("DONE.")