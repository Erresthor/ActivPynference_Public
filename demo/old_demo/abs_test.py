#!/usr/bin/python
from operator import index
from statistics import variance
import sys,inspect,os
import numpy as np
import matplotlib.pyplot as plt
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from pyai.base.file_toolbox import save_flexible,load_flexible
from pyai.layer.layer_learn import MemoryDecayType
from pyai.model.active_model import ActiveModel
from pyai.neurofeedback_run import save_model_performance_dictionnary
from pyai.neurofeedback_run import trial_plot_from_name
#from pyai.models_neurofeedback.climb_stairs import nf_model,evaluate_container

from pyai.models_neurofeedback.article_1_simulations.climb_stairs_flat_priors import nf_model_imp2,evaluate_container
# Generate a succession of trial results for a model in the list generated
# Made to parrallelize in a cluster-like environment

if __name__=="__main__":
    # Grid of prior values explored : 


    save_path = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","TEMPORARY_TEST_BED")
    model_name = "2_a_poor_true_feedback"
    Ninstances = 10
    Ntrials = 20
    overwrite = False

    parameters = [True,"perfect",1,True,1.0,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000]
    parameters = [True,2.0,1,True,1.0,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000]
    var = True
    comp = True # By default, save everything ^^
    
    # Get the index-th element of the dictionnary to generate the related 
    a_learn = parameters[0]
    perfect_a = (parameters[1]=="perfect")
    if perfect_a :
        a_acc = 0.0
    else :
        a_acc = float(parameters[1])
    a_str = parameters[2]
    
    b_learn = parameters[3]
    perfect_b = (parameters[4]=="perfect")
    if perfect_b :
        b_acc = 0.0
    else :
        b_acc = float(parameters[4])
    b_str = parameters[5]
    
    d_learn = parameters[6]
    memory_decay_type = parameters[7]
    memory_decay_halftime = parameters[8]

    prop_poubelle = 0.0

    # THIS GENERATES THE TRIALS without using run_model
    model = nf_model_imp2(model_name,save_path,prop_poubelle = prop_poubelle,
                        learn_a = a_learn,prior_a_ratio = a_acc,prior_a_strength=a_str,
                        learn_b=b_learn,prior_b_ratio = b_acc,prior_b_strength=b_str,
                        learn_d=d_learn,
                        mem_dec_type=memory_decay_halftime,mem_dec_halftime=memory_decay_halftime,
                        perfect_a = perfect_a,perfect_b=perfect_b,verbose = False)
    
    
    model.input_parameters = parameters
    model.index = 0 # This is to track the input parameters
    # We can also add custom values for the matrices or modify the run here :
    # model.parameter = new_value
    model.initialize_n_layers(Ninstances)
    model.run_n_trials(Ntrials,overwrite=overwrite,global_prop=None)

    # Here, we shoud generate a first set of run-wide performance results
    # And save them in a dedicated file (_MODEL and _PERFORMANCES or smthg like that ?)
    save_model_performance_dictionnary(save_path,model_name,evaluate_container,overwrite=overwrite,include_var=var,include_complete=comp)
    print("Saving model results at :   " + save_path + " " + model_name)
    trial_plot_from_name(save_path,model_name,3,[0,1,2,10,19])
    input()