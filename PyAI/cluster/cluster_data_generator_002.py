#!/usr/bin/python
from operator import index
import sys,inspect,os
from imageio import save
import numpy as np
import matplotlib.pyplot as plt
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from pyai.base.file_toolbox import save_flexible,load_flexible
from pyai.layer.layer_learn import MemoryDecayType
from pyai.model.active_model import ActiveModel
from pyai.neurofeedback_run import evaluate_model_mean,evaluate_model_dict

#from pyai.models_neurofeedback.climb_stairs import nf_model,evaluate_container

from pyai.models_neurofeedback.article_1_simulations.climb_stairs_002 import nf_model,evaluate_container
# Generate a succession of trial results for a model in the list generated
# Made to parrallelize in a cluster-like environment

# Grid of prior values explored : 
prior_value_a_sigma = np.arange(0,5.25,0.25)
prior_value_b_sigma = np.arange(0,5.25,0.25)

def generate_a_parameter_list(a_priors,b_priors) :
    # Undordered dictionnaries are soooo not cool :(
    new_list = []
    indexlist = [] #This is the list of how the nth model would be situated in a k-dim grid
    for ka in range(a_priors.shape[0]):
        for kb in range(b_priors.shape[0]):
            modelchar = [False,a_priors[ka],1,True,b_priors[kb],1,True,MemoryDecayType.NO_MEMORY_DECAY,2000]
            modelname = "a_ac"+str(int(100*a_priors[ka]))+"_str1_b_ac"+str(int(100*b_priors[kb]))+"_str1"
            new_list.append([modelname,modelchar])
            indexlist.append([ka,kb])
    return new_list,indexlist

parameter_list,index_list = generate_a_parameter_list(prior_value_a_sigma,prior_value_b_sigma)

if __name__=="__main__":
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
        model_name = parameter_list[list_index][0]
        model_options = parameter_list[list_index][1]
    except :
        print("Index beyond model list length")
        sys.exit()
    
    # Get the index-th element of the dictionnary to generate the related 
    a_learn = model_options[0]
    a_acc = max(model_options[1],0.005)
    a_str = model_options[2]
    b_learn = model_options[3]
    b_acc = max(model_options[4],0.005)
    b_str = model_options[5]
    d_learn = model_options[6]
    memory_decay_type = model_options[7]
    memory_decay_halftime = model_options[8]
    # THIS GENERATES THE TRIALS without using run_model
    model = nf_model(model_name,save_path,prop_poubelle=0.0,prior_a_sigma=a_acc,prior_a_strength=a_str,learn_a=a_learn,
                                                        prior_b_sigma=b_acc,prior_b_strength=b_str,learn_b=b_learn,
                                                        learn_d=d_learn,
                                                        mem_dec_type=memory_decay_type,mem_dec_halftime=memory_decay_halftime,
                                                        verbose=False)
    
    
    model.input_parameters = model_options
    model.index = index_list[list_index] # This is to track the input parameters
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