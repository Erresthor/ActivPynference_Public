import sys,inspect,os
from pickle import FALSE
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
from pyai.neurofeedback_run import evaluate_model_mean



def produce_sumup_for(savepath,evaluator):
    # We're in a folder where a lot of models can be stored all next to each other. Let's go through them all !
    all_folders = [f for f in os.listdir(savepath) if (os.path.isdir(os.path.join(savepath, f)))]
    output_list = []
    for model in all_folders:
        output_list.append(produce_model_sumup_for(model,savepath,evaluator))
    save_flexible(output_list,os.path.join(savepath,"simulation_output.pyai"))
    return output_list

def produce_model_sumup_for(modelname,savepath,evaluator):
    # There is a MODEL_ file here,we should grab it and extract the model inside, it is a gound indicator of the input parameters
    # for this run !
    model_path = os.path.join(savepath,modelname)
    model_object = ActiveModel.load_model(model_path)
    
    # There are also instances here, we should generate mean indicators to get the general performances!
    mean_A,mean_B,mean_D,a_err_arr,b_err_arr,Ka_arr,Kb_arr,Kd_arr,error_states_arr,error_behaviour_arr,total_instances = evaluate_model_mean(evaluator,modelname,savepath)
    # # We can manually go through the instances :
    # all_instances = [f for f in os.listdir(model_path) if (os.path.isdir(os.path.join(model_path, f)))]
    # for instance in all_instances :
    #     instance_path = os.path.join(model_path,instance)
    performance_list = [mean_A,mean_B,mean_D,a_err_arr,b_err_arr,Ka_arr,Kb_arr,Kd_arr,error_states_arr,error_behaviour_arr,total_instances]
    model_list = [model_object,performance_list]
    return model_list

if __name__=="__main__" :
    # AFAIK, there is no way to check if cluster tasks are over. This has to be launched manually to generate a sumup matrix file.
    # We then send it (by mail ?) to the local post for analysis
    input_arguments = sys.argv
    assert len(input_arguments)>=2,"Data generator needs at least 1 argument : savepath"
    name_of_script = input_arguments[0]
    save_path = input_arguments[1]

    # prior_value_a = prior_value_a
    # prior_value_b = prior_value_b
    # parameter_list = parameter_list # This is the parameter list from the cluster data generator (list of [model_name, model_options])

    bigbig_list = produce_sumup_for(save_path,evaluate_container)
    print("Saving output to  -  " + save_path + "simulation_output.pyai")
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