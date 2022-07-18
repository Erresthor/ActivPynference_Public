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

from pyai.neurofeedback_run import run_models,run_model,evaluate_model,evaluate_model_mean,trial_plot_from_name,evaluate_model_dict

#from pyai.models_neurofeedback.climb_stairs import nf_model,evaluate_container

from pyai.models_neurofeedback.article_1_simulations.climb_stairs_002 import nf_model,evaluate_container

# Generate a succession of trial results for a model in the list generated
# Made to parrallelize in a cluster-like environment

# Grid of prior values explored : 
prior_value_a_sigma = np.array([0.05,0.1,0.2,0.3,0.5,1.0,2.0,3.5,5.0,7.5,10.0])
prior_value_b_sigma = np.array([0.05,0.1,0.2,0.3,0.5,1.0,2.0,3.5,5.0,7.5,10.0])

def generate_a_parameter_list(a_priors,b_priors) :
    # Undordered dictionnaries are soooo not cool :(
    new_list = []
    indexlist = [] #This is the list of how the nth model would be situated in a k-dim grid
    for ka in range(a_priors.shape[0]):
        for kb in range(b_priors.shape[0]):
            modelchar = [True,a_priors[ka],1,True,b_priors[kb],1,True,MemoryDecayType.STATIC,1000]
            modelname = "a_ac"+str(int(10*a_priors[ka]))+"_str1_b_ac"+str(int(10*b_priors[kb]))+"_str1"
            new_list.append([modelname,modelchar])
            indexlist.append([ka,kb])
    return new_list,indexlist

def save_model_sumup_for(modelname,savepath,evaluator):
    # There is a MODEL_ file here,we should grab it and extract the model inside, it is a gound indicator of the input parameters
    # for this run !
    model_path = os.path.join(savepath,modelname)
    model_object = ActiveModel.load_model(model_path)
    
    # There are also instances here, we should generate mean indicators to get the general performances!
    mean_A,mean_B,mean_D,a_err_arr,b_err_arr,Ka_arr,Kb_arr,Kd_arr,error_states_arr,error_behaviour_arr,error_observations_arr,error_perception_arr,total_instances = evaluate_model_mean(evaluator,modelname,savepath)
    # # We can manually go through the instances :
    # all_instances = [f for f in os.listdir(model_path) if (os.path.isdir(os.path.join(model_path, f)))]
    # for instance in all_instances :
    #     instance_path = os.path.join(model_path,instance)
    performance_list = [mean_A,mean_B,mean_D,a_err_arr,b_err_arr,Ka_arr,Kb_arr,Kd_arr,error_states_arr,error_behaviour_arr,error_observations_arr,error_perception_arr,total_instances]
    model_list = [model_object,performance_list]
    return model_list

parameter_list,index_list = generate_a_parameter_list(prior_value_a_sigma,prior_value_b_sigma)

if __name__=="__main__":
    # ENVIRONMENT
    overwrite = True
    save_path = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","TEMPORARY_TEST_BED")
    model_name = "including_d_error_2"

    list_index = 120
    list_index = 5
    Ninstances = 5
    Ntrials = 10

    modelchar = [True,0.01,1,True,50,1,True,MemoryDecayType.NO_MEMORY_DECAY,1000]
    modelname = "a_ac"+str(int(100*modelchar[1]))+"_str1_b_ac"+str(int(100*modelchar[4]))+"_str1"
    model_options = modelchar
    # try :
    #     model_options = parameter_list[list_index][1]
    # except :
    #     print("Index beyond model list length")
    #     sys.exit()
    
    # Get the index-th element of the dictionnary to generate the related 
    a_learn = model_options[0]
    a_acc = model_options[1]
    a_str = model_options[2]
    b_learn = model_options[3]
    b_acc = model_options[4]
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
    
    print(np.round(model.a[0],2))
    print("----------")
    for action in range(5):
        print(np.round(model.b[0][:,:,action],2))
    model.input_parameters = model_options
    model.index = index_list[list_index] # This is to track the input parameters
    # We can also add custom values for the matrices or modify the run here :
    # model.parameter = new_value

    model.initialize_n_layers(Ninstances)
    trial_times = [0.01]
    model.verbose = False
    #model.run_n_trials(Ntrials,overwrite=overwrite,global_prop=None,list_of_last_n_trial_times=trial_times)

    # Old version : (could not shunt the way model options impacted the model :/ too restrictive or a better way of following things up ?)
    # run_model(save_path,model_name,model_options,Ntrials,Ninstances,overwrite = False,global_prop=[0,1],verbose=False)
    

    # Here, we shoud generate a first set of run-wide performance results
    # And save them in a dedicated file (_MODEL and _PERFORMANCES or smthg like that ?)
    

    #generate_model_sumup(model_name,save_path,True)
    # sumup  = save_model_sumup_for(model_name,save_path,evaluate_container)[1]

    dict_sumup = evaluate_model_dict(evaluator=evaluate_container,modelname=model_name,savepath=save_path)
    # mean_A,mean_B,mean_D,a_err_arr,b_err_arr,Ka_arr,Kb_arr,Kd_arr,error_states_arr,error_behaviour_arr,error_observations_arr,error_perception_arr,total_instances
    # 0      1       2        3           4     5       6    7         8                  9                      10                 11  
    
    print(dict_sumup['variance'])
    # general_performance_plot(save_path,model_name,"GLOBAL",
    #         np.arange(0,Ntrials,1),sumup[3],sumup[4],sumup[5],sumup[6],sumup[8],sumup[9])
    

    # trial_plot_from_name(save_path,model_name,0,[0,1,2,3,Ntrials-2,Ntrials-1])
    # plt.show()
    # prior_value_a_sigma = np.arange(0,5.25,0.25)
    # print(prior_value_a_sigma)