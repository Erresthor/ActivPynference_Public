#!/usr/bin/python
from operator import index
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
from pyai.neurofeedback_run import trial_plot, trial_plot_from_name
from pyai.models_neurofeedback.article_1_simulations.climb_stairs_b_flat_a_gaussian import nf_model,evaluate_container

# Generate a succession of trial results for a model in a list
# Made to work as a command and simulate a single model index
# To integrate in a cluster like environment

def generate_a_parameter_list(a_priors,b_priors) :
    # Undordered dictionnaries are soooo not cool :(
    new_list = []
    indexlist = [] #This is the list of how the nth model would be situated in a k-dim grid
    for ka in range(len(a_priors)):
        for kb in range(len(b_priors)):
            modelchar = [True,a_priors[ka],10,True,b_priors[kb],1,True,MemoryDecayType.NO_MEMORY_DECAY,2000]
            
            try :
                labela = str(int(100*a_priors[ka]))
            except : 
                labela = str(a_priors[ka])
            
            try :
                labelb = str(int(100*b_priors[kb]))
            except : 
                labelb = b_priors[kb]
            
            modelname = "a_ac"+labela+"_str10_b_ac"+labelb+"_str1"
            new_list.append([modelname,modelchar])
            indexlist.append([ka,kb])
    return new_list,indexlist

if __name__=="__main__":
    prior_value_a_sigma = list(np.arange(0,5.25,0.25))
    prior_value_a_sigma.append("perfect")
    prior_value_b_sigma = list(np.arange(0,5.25,0.25))
    prior_value_b_sigma.append("perfect")

    parameter_list,index_list = generate_a_parameter_list(prior_value_a_sigma,prior_value_b_sigma)

    
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
        perf_indicators = input_arguments[6]
        assert ('m' in perf_indicators), "mean should be among perf indicators"
        var = ('v' in perf_indicators)
        comp = ('c' in perf_indicators)
    except :
        var = True
        comp = True # By default, save everything ^^

    try :
        model_name = parameter_list[list_index][0]
        model_options = parameter_list[list_index][1]
    except :
        print("Index beyond model list length")
        sys.exit()
    
    # Get the index-th element of the dictionnary to generate the related 
    # Get the index-th element of the dictionnary to generate the related 
    a_learn = model_options[0]
    perfect_a = (model_options[1]=="perfect") # In this case, perfect means the feedback is perfect
    if perfect_a :
        a_acc = 0.01
    else :
        a_acc = max(model_options[1],0.005)
    a_str = model_options[2]
    
    b_learn = model_options[3]
    perfect_b = (model_options[4]=="perfect")
    if perfect_b :
        b_acc = 0.0
    else :
        b_acc = model_options[4]
    b_str = model_options[5]

    d_learn = model_options[6]
    memory_decay_type = model_options[7]
    memory_decay_halftime = model_options[8]

    prop_poubelle = 0.0

    # THIS GENERATES THE TRIALS without using run_model
    model = nf_model(model_name,save_path,prop_poubelle=prop_poubelle,
                    prior_a_sigma=a_acc,prior_a_strength=a_str,learn_a=a_learn,
                    prior_b_ratio=b_acc,prior_b_strength=b_str,learn_b=b_learn,
                    learn_d=d_learn,
                    mem_dec_type=memory_decay_type,mem_dec_halftime=memory_decay_halftime,
                    verbose=False,perfect_a=perfect_a,perfect_b=perfect_b,SHAM="True")
    
    
    model.input_parameters = model_options
    model.index = index_list[list_index] # This is to track the input parameters
    # We can also add custom values for the matrices or modify the run here :
    # model.parameter = new_value
    model.initialize_n_layers(Ninstances)
    model.run_n_trials(Ntrials,overwrite=overwrite,global_prop=None)

    # Here, we shoud generate a first set of run-wide performance results
    # And save them in a dedicated file (_MODEL and _PERFORMANCES or smthg like that ?)
    save_model_performance_dictionnary(save_path,model_name,evaluate_container,overwrite=overwrite,include_var=var,include_complete=comp)
    print("Saving model results at :   " + save_path + " " + model_name)

