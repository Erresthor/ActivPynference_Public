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
from pyai.models_neurofeedback.article_1_simulations.climb_stairs_a_gaussian_A_gaussian import nf_model,evaluate_container

# Generate a succession of trial results for a model in a list
# Made to work as a command and simulate a single model index
# To integrate in a cluster like environment

def generate_a_parameter_list(a_priors_sigma,a_priors_mean,A_priors_sigma,A_priors_mean) :
    # Undordered dictionnaries are soooo not cool :(
    new_list = []
    indexlist = [] #This is the list of how the nth model would be situated in a k-dim grid
    for ka in range(len(a_priors_sigma)):
        for kA in range(len(A_priors_sigma)):
            for ma in range(len(a_priors_mean)):
                for mA in range(len(A_priors_mean)):
                    modelchar = [True    ,a_priors_sigma[ka] ,a_priors_mean[ma]    ,A_priors_sigma[kA] ,A_priors_mean[mA]]
                    #            learn a ,a_sigma      ,a_mean ,A_sigma      ,A_mean
                    try :
                        labela1 = str(int(100*a_priors_sigma[ka]))
                    except : 
                        labela1 = str(a_priors_sigma[ka])
                        
                    try :
                        labela2 = str(int(100*a_priors_mean[ma]))
                    except : 
                        labela2 = str(a_priors_mean[ma])
                    
                    try :
                        labelA1 = str(int(100*A_priors_sigma[kA]))
                    except : 
                        labelA1 = str(A_priors_sigma[kA])
                        
                    try :
                        labelA2 = str(int(100*A_priors_mean[mA]))
                    except : 
                        labelA2 = str(A_priors_mean[mA])
                    
                    modelname = "a_"+labela1+"_"+ labela2+ "_A_"+labelA1+"_"+labelA2
                    new_list.append([modelname,modelchar])
                    indexlist.append([ka,kA,ma,mA])
    return new_list,indexlist

if __name__=="__main__":
    a_sigma = list(np.arange(0,2,0.25))
    a_mean = list(np.arange(-1,1,0.25))

    print(a_sigma)
    print(a_mean)

    #A_sigma = list(np.arange(0,3,0.25))
    #A_mean = list(np.arange(-1,1,0.1))

    A_sigma = ["perfect"]
    A_mean = [0]

    parameter_list,index_list = generate_a_parameter_list(a_sigma,a_mean,A_sigma,A_mean)

    
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
        a_sigma = 0.01
    else :
        a_sigma = max(model_options[1],0.005)
    a_mean = model_options[2]
    
    perfect_A = (model_options[3]=="perfect") 
    if perfect_A:
        A_sigma = 0.01
    else: 
        A_sigma = model_options[3]
    A_mean = model_options[4]

    prop_poubelle = 0.0

    # THIS GENERATES THE TRIALS without using run_model
    model = nf_model(model_name,save_path,prop_poubelle=prop_poubelle,
                        prior_a_sigma=a_sigma,prior_a_meanskew=a_mean,learn_a=a_learn,
                        prior_A_sigma=A_sigma,prior_A_meanskew=A_mean,
                        perfect_a=perfect_a,perfect_A=perfect_A)
    
    
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

    trial_plot_from_name(save_path,model_name,2,[1,4,9])
    plt.show()
    print(model_name)
    print(model_options)
    
    print(np.round(model.a,2))
    print(np.round(model.A,2))