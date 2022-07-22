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
from pyai.neurofeedback_run import run_models
from pyai.neurofeedback_run import evaluate_model_mean,evaluate_model_dict


def produce_model_sumup_for(modelname,savepath,evaluator):
    # There is a MODEL_ file here,we should grab it and extract the model inside, it is a gound indicator of the input parameters
    # for this run !
    model_path = os.path.join(savepath,modelname)
    model_object = ActiveModel.load_model(model_path)
    
    # There are also instances here, we should generate mean indicators to get the general performances!
    # mean_A,mean_B,mean_D,a_err_arr,b_err_arr,Ka_arr,Kb_arr,Kd_arr,error_states_arr,error_behaviour_arr,error_observations_arr,error_perception_arr,total_instances = evaluate_model_mean(evaluator,modelname,savepath)
    # 0      1      2      3         4         5       6      7            8             9                   10                     11
    
    model_evaluator_dictionnary = evaluate_model_dict(evaluator,modelname,savepath)

    complete_evaluator = model_evaluator_dictionnary['complete']
    mean_evaluator = model_evaluator_dictionnary['mean']
    variance_evaluator = model_evaluator_dictionnary['variance']
    # # We can manually go through the instances :
    # all_instances = [f for f in os.listdir(model_path) if (os.path.isdir(os.path.join(model_path, f)))]
    # for instance in all_instances :
    #     instance_path = os.path.join(model_path,instance)
    model_dict = {
        'model':model_object,
        'mean':mean_evaluator,
        'variance':variance_evaluator
    }

    return model_dict,model_object

def produce_total_sumup_for(savepath,evaluator,overwrite=False,keyword='mean'):
    # We're in a folder where a lot of models can be stored all next to each other. Let's go through them all !
    all_folders = [f for f in os.listdir(savepath) if (os.path.isdir(os.path.join(savepath, f)))]
    output_list = []
    for model in all_folders:
        modelpath = os.path.join(savepath,model)
        # Check if the file with local performances as been generated :
        
        # MEAN
        if(keyword=='mean') :
            potential_file = os.path.join(modelpath,"_PERFORMANCES_MEAN")
            print("------   " + model + " -------")
            if (os.path.isfile(potential_file))and(not(overwrite)) :
                local_sumup = load_flexible(potential_file)
            else : # Else, we generate it here. It is suboptimal because we do not parrallelize this operations (=/= cluster)
                local_sumup = produce_model_sumup_for(model,savepath,evaluator)
                save_flexible(local_sumup,potential_file)
            output_list.append(local_sumup)
        elif(keyword=='var') :
            potential_file = os.path.join(modelpath,"_PERFORMANCES_VAR")
            print("------   " + model + " -------")
            if (os.path.isfile(potential_file))and(not(overwrite)) :
                local_sumup = load_flexible(potential_file)
            else : # Else, we generate it here. It is suboptimal because we do not parrallelize this operations (=/= cluster)
                local_sumup = produce_model_sumup_for(model,savepath,evaluator)
                save_flexible(local_sumup,potential_file)
            output_list.append(local_sumup)
    return output_list


if __name__=="__main__" :
    # Come ooon
    # AFAIK, there is no way to check if cluster tasks are over. This has to be launched manually to generate a sumup matrix file.
    # We then send it (by mail ?) to the local post for analysis
    input_arguments = sys.argv
    assert len(input_arguments)>=2,"Data generator needs at least 1 argument : savepath"
    name_of_script = input_arguments[0]
    save_path = input_arguments[1]
    try : 
        overwrite = input_arguments[2]
        overwrite = (overwrite=="True")
    except :
        overwrite = False
    
    print("------------------------------------------------------------------")
    print("Generating sumup for " + save_path)
    if(overwrite):
        print("(Overwriting previous files)")
    print("------------------------------------------------------------------")
    list_of_models_mean = produce_total_sumup_for(save_path,evaluate_container,overwrite=overwrite,keyword='mean')
    list_of_models_var = produce_total_sumup_for(save_path,evaluate_container,overwrite=overwrite,keyword='var')

    sumup_file_name_mean = "simulation_output_" + save_path.split("_")[-1] + "_mean" + ".pyai"
    sumup_file_name_var = "simulation_output_" + save_path.split("_")[-1] + "_var" + ".pyai"

    save_flexible(list_of_models_mean,os.path.join(save_path,sumup_file_name_mean))
    save_flexible(list_of_models_var,os.path.join(save_path,sumup_file_name_var))

    print("Saving output to   -  " + save_path + sumup_file_name_mean + " / " + sumup_file_name_var)
    # Multimodel plot :
    # action_errors = []
    # state_errors = []
    # a_model_errors = []
    # b_model_errors = []

    # t = np.arange(0,Ntrials,1)
    # # arr  = (np.array(all_stat_err))

    # savenam = os.path.join(save_path,"output_array.my_arr")
    # # save_flexible(arr,savenam)

    # arr = load_flexible(savenam)
    # # print(arr.shape)
    # # for i in range(len(models_dictionnary)):
    # #     y = arr[i,:]
    # #     if y.shape[0]>Ntrials:
    # #         y = y[:Ntrials]
    # #     plt.scatter(t,y,s=1)

    # # # Single value of a or b

    # # for i in range(len(models_dictionnary)):
    # #     y = arr[i,:]
    # #     t = np.arange(0,Ntrials,1)
    # #     y_av = sliding_window_mean(list(y),4)
    # #     y_av = np.array(y_av)
    # #     if y_av.shape[0]>Ntrials:
    # #         y_av = y_av[:Ntrials]
        
    # #     my_key = list(models_dictionnary)[i]
    # #     list_of_key =  (my_key.split("_"))
    # #     a_acc = float(list_of_key[1].strip("ac"))/10
    # #     b_acc = float(list_of_key[4].strip("ac"))/10
    # #     print(a_acc,b_acc)
    # #     prior_value = models_dictionnary[list(models_dictionnary)[i]][1]
    # #     if (a_acc == 1.0) or (a_acc==1.5):
    # #         plt.plot(t,y_av,label="Good Prior Biais =  " + str(prior_value) + " b_acc = " + str(b_acc))

    # # plt.legend()
    # # plt.xlabel("Trials")
    # # plt.ylabel("State error w.r.t optimal")
    # # plt.title("How does prior influence overall performance")
    # # plt.grid(True)
    # # plt.show()

    # # 3D plot


    # the_t = 100
    # xs = []
    # ys = []
    # zs = []

    # icnt = 0
    # jcnt = 0
    # J = prior_value_a.shape[0]
    # plot_this = np.zeros((J,J))

    # for i in range(len(models_dictionnary)):
    #     y = arr[i,:]
    #     t = np.arange(0,Ntrials,1)
    #     y_av = sliding_window_mean(list(y),10)
    #     y_av = np.array(y_av)
    #     if y_av.shape[0]>Ntrials:
    #         y_av = y_av[:Ntrials]
        
    #     my_key = list(models_dictionnary)[i]
    #     list_of_key =  (my_key.split("_"))
    #     a_acc = float(list_of_key[1].strip("ac"))/10
    #     b_acc = float(list_of_key[4].strip("ac"))/10
    #     #if (a_acc < 50)and(b_acc<50):
    #     xs.append(a_acc)
    #     ys.append(b_acc)
    #     zs.append(y_av[the_t])
    #     plot_this[icnt,jcnt] = y_av[the_t]

    #     icnt = icnt+1
    #     if (icnt>=J):
    #         icnt = 0
    #         jcnt = jcnt+1



    # from matplotlib import cm
    # print(plot_this)
    # zs = np.array(zs)
    # X,Y = np.meshgrid(prior_value_a,prior_value_a)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(xs, ys, zs, cmap='Greens')
    # ax.plot_surface(X,Y,plot_this,linewidth=0,cmap=cm.coolwarm, antialiased=False)
    # plt.show()