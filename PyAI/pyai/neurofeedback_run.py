from pickle import FALSE
import numpy as np
import os
import matplotlib.pyplot as plt

from .base.file_toolbox import save_flexible,load_flexible

from .layer.layer_learn import MemoryDecayType
from .base.function_toolbox import normalize

from .model.active_model import ActiveModel
from .model.active_model_save_manager import ActiveSaveManager
from .model.active_model_container import ActiveModelSaveContainer


from .model.model_visualizer import belief_matrices_plots,generate_model_sumup,general_performance_plot,trial_plot_figure

from .models_neurofeedback.climb_stairs import nf_model,evaluate_container

# UNIQUE FOR THIS MODEL !!


# EXTRACTORS OF TRIALS
def evaluate_trial(evaluator,trialcontainer,a_err,b_err,Ka,Kb,Kd,error_states,error_behaviour):
    eval_cont = evaluator(trialcontainer)

    a_err.append(eval_cont['a_dir'])
    b_err.append(eval_cont['b_dir'])
    Ka.append(eval_cont['a_uncertainty'])
    Kb.append(eval_cont['b_uncertainty'])
    Kd.append(eval_cont['d_uncertainty'])

    error_states.append(eval_cont['mean_error_state'])
    error_behaviour.append(eval_cont['mean_error_behaviour'])

def evaluate_instance(evaluator,savepath,modelname,instance_number=0,return_matrices=False):
    instance_string = f'{instance_number:03d}'
    instance_folder = os.path.join(savepath,modelname,instance_string)

    total_trials = len(os.listdir(instance_folder))#number_of_trials_in_instance_folder(instance_folder)    
    trials = range(total_trials)
    
    Ka,Kb,Kd,a_err,b_err,error_states,error_behaviour = [],[],[],[],[],[],[]
    if(return_matrices):
        A_list,B_list,D_list = [],[],[]
    
    for trial in trials:
        cont = ActiveSaveManager.open_trial_container(os.path.join(savepath,modelname),instance_number,trial,'f')
        evaluate_trial(evaluator,cont,a_err,b_err,Ka,Kb,Kd,error_states,error_behaviour)
        if(return_matrices):
            try :
                a_mat = cont.a_
            except :
                a_mat = cont.A_
            A_list[-1].append(a_mat)

            try :
                b_mat = cont.b_
            except :
                b_mat = cont.B_
            B_list[-1].append(b_mat)

            try :
                d_mat = cont.d_
            except :
                d_mat = cont.D_
            D_list[-1].append(d_mat)
    
    if (return_matrices):
        return trials,a_err,b_err,Ka,Kb,Kd,error_states,error_behaviour,A_list,B_list,D_list
    else :
        return trials,a_err,b_err,Ka,Kb,Kd,error_states,error_behaviour
 
def evaluate_model(evaluator,modelname,savepath) : 
    """Return all the performance indicators implemented for a given model accross all layer instances
    TODO : reimplement using the "general performance indicators" function !"""
    loadpath = os.path.join(savepath,modelname)

    A_list,B_list,D_list = [],[],[]
    
    Ka,Kb,Kd,a_err,b_err,error_states,error_behaviour = [],[],[],[],[],[],[]        
    model = ActiveModel.load_model(loadpath)

    for potential_instance in os.listdir(loadpath):
        complete_path = os.path.join(loadpath,potential_instance)
        is_dir = (os.path.isdir(complete_path))
        if (is_dir) : # Only files are instances
            if ("_RESULTS" in potential_instance) or ("_MODEL" in potential_instance): 
                # Wait, this is no instance ! >:(
                # --> ignore this file 
                print("Ignoring file " + potential_instance)
                continue

            # This seems to be an instance :D
            print("Adding instance  : "+ potential_instance + " to the mean trial.")

            # This is trial results (layer instance)
            layer_instance = int(potential_instance)

            trials,a_err_i,b_err_i,Ka_i,Kb_i,Kd_i,error_states_i,error_behaviour_i,A_list_i,B_list_i,D_list_i = evaluate_instance(evaluator,savepath,modelname,instance_number=layer_instance,return_matrices=True)
            
            A_list.append(A_list_i)
            B_list.append(B_list_i)
            D_list.append(D_list_i)
            Ka.append(Ka_i)
            Kb.append(Kb_i)
            Kd.append(Kd_i)
            a_err.append(a_err_i)
            b_err.append(b_err_i)
            error_states.append(error_states_i)
            error_behaviour.append(error_behaviour_i)

    Ka_arr = np.array(Ka)
    Kb_arr = np.array(Kb)
    Kd_arr = np.array(Kd)
    a_err_arr = np.array(a_err)
    b_err_arr = np.array(b_err)
    error_states_arr = np.array(error_states)
    error_behaviour_arr = np.array(error_behaviour)

    return A_list,B_list,D_list,Ka_arr,Kb_arr,Kd_arr,a_err_arr,b_err_arr,error_states_arr,error_behaviour_arr

def mean_indicators(A_list,B_list,D_list,Ka_arr,Kb_arr,Kd_arr,a_err_arr,b_err_arr,error_states_arr,error_behaviour_arr):
    def mean_over_first_dim(list_of_list_of_matrices):
        def flexible_sum(list_of_matrices_1,list_of_matrices_2):
            assert len(list_of_matrices_1)==len(list_of_matrices_2),"List should be equal dimensions before summing"
            r = []
            for k in range(len(list_of_matrices_1)) :
                r.append(list_of_matrices_1[k] + list_of_matrices_2[k])
            return r
        
        r = [0 for i in range(len(list_of_list_of_matrices[0]))]
        cnt = 0
        for list_of_matrices in list_of_list_of_matrices :
            r = flexible_sum(r, list_of_matrices)
            cnt = cnt + 1.0

        # Mean :
        for k in range(len(list_of_list_of_matrices[0])):
            r[k] = r[k]/cnt
        
        return r

    mean_A = []
    mean_B = []
    mean_D = []
    total_instances = len(A_list)
    for t in range(len(A_list[0])): # Iterating through timesteps
        a_at_t = []
        b_at_t = []
        d_at_t = []
        for k in range(len(A_list)):
            a_at_t.append(normalize(A_list[k][t]))
            b_at_t.append(normalize(B_list[k][t]))
            d_at_t.append(normalize(D_list[k][t]))

        mean_A.append(mean_over_first_dim(a_at_t))
        mean_B.append(mean_over_first_dim(b_at_t))
        mean_D.append(mean_over_first_dim(d_at_t))

    Ka_arr = np.mean(Ka_arr,axis=0)
    Kb_arr = np.mean(Kb_arr,axis=0)
    Kd_arr = np.mean(Kd_arr,axis=0)
    a_err_arr = np.mean(a_err_arr,axis=0)
    b_err_arr = np.mean(b_err_arr,axis=0)
    error_states_arr = np.mean(error_states_arr,axis=0)
    error_behaviour_arr = np.mean(error_behaviour_arr,axis=0)

    return mean_A,mean_B,mean_D,a_err_arr,b_err_arr,Ka_arr,Kb_arr,Kd_arr,error_states_arr,error_behaviour_arr,total_instances

def evaluate_model_mean(evaluator,modelname,savepath) :
    """Generate the mean trial by selecting the mean value accross all instances for every matrix and error estimators    """
    A_list,B_list,D_list,Ka_arr,Kb_arr,Kd_arr,a_err_arr,b_err_arr,error_states_arr,error_behaviour_arr = evaluate_model(evaluator,modelname,savepath)
    return mean_indicators(A_list,B_list,D_list,Ka_arr,Kb_arr,Kd_arr,a_err_arr,b_err_arr,error_states_arr,error_behaviour_arr)

def generate_instances_figures(evaluator,savepath,modelname,instance_list,gifs=False,mod_ind=0,fac_ind=0,show=False):
    """ Plot of individual agent performances over all instances for a given model."""
    generate_model_sumup(modelname,savepath,gifs,mod_ind,fac_ind) # Matrices and gifs to see how the training went
    for inst in instance_list:
        generate_instance_performance_figure(evaluator,savepath,modelname,inst,show=show) # A single plot encompassing matrix error & behaviour optimality

def generate_instance_performance_figure(evaluator,savepath,modelname,instance_number=0) :
    """ Plot of a and b knowledge error + state and behaviour error."""
    trials,a_err,b_err,Ka,Kb,Kd,error_states,error_behaviour = evaluate_instance(evaluator,savepath,modelname,instance_number)
    save_string = f'{instance_number:03d}'
    figtitle = modelname +" - Instance " + str(instance_number) + " performance sumup"
    general_performance_plot(savepath,modelname,save_string,trials,a_err,b_err,Ka,Kb,error_states,error_behaviour,smooth_window = 5,show=False,figtitle=figtitle)

def evaluate_model_figure(evaluator,savepath,modelname,show=True):
    """ generate_instances_figure but for an hypothetical """
    mean_A,mean_B,mean_D,a_err,b_err,Ka_arr,Kb_arr,Kd_arr,error_states_arr,error_behaviour_arr,tot_instances = evaluate_model_mean(evaluator,modelname,savepath)
    n = a_err.shape[0]
    trials = np.linspace(0,n,n)
    
    general_performance_plot(savepath,modelname,"GLOBAL",trials,a_err,b_err,Ka_arr,Kb_arr,error_states_arr,error_behaviour_arr,smooth_window=5,figtitle=modelname+" - performance sumup over " + str(tot_instances) + " instance(s)",show=True)
    belief_matrices_plots(modelname,savepath,mean_A,mean_B,mean_D,plot_gifs=True)

def trial_plot(savecontainer,plotmean=False,action_labels="alphabet",title=None,
                hidden_state_factor = 0,perc_modality = 0):
    T = savecontainer.T
    
    obs = savecontainer.o[perc_modality,:]
    states = savecontainer.s[hidden_state_factor,:]
    acts = savecontainer.u[hidden_state_factor,:]
    beliefs = savecontainer.X[hidden_state_factor]
    u_post = savecontainer.U_post

    # BEGIN ! --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
    try :
        a_mat = savecontainer.a_[perc_modality]
    except:
        a_mat = savecontainer.A_[perc_modality]
    while (a_mat.ndim < 3):
        a_mat = np.expand_dims(a_mat,-1)
    
    try :
        b_mat = savecontainer.b_[hidden_state_factor]
    except :
        b_mat = savecontainer.B_[hidden_state_factor]
    
    figure = trial_plot_figure(T,beliefs,u_post,
                obs,states,acts,
                a_mat,b_mat,
                plotmean=plotmean,action_labels=action_labels,title=title)
    figure.show()

def run_a_trial():
    """An example trial to check if the whole package actually works"""
    # ENVIRONMENT
    savepath = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","code","results","single_models")
    modelname = "standard"

    # SIMULATING TRAINING
    model = nf_model(modelname,savepath,prop_poubelle=0.3,prior_a_ratio=10,prior_a_strength=2,prior_b_ratio=1,prior_b_strength=1)
    Ninstances = 10
    trials_per_instances = 250
    model.initialize_n_layers(Ninstances)
    overwrite = True
    model.run_n_trials(trials_per_instances,overwrite=overwrite)

    # FIGURES AND ANALYSIS
    instance_list = [i for i in range(Ninstances)]
    modality_indice = 0
    factor_indice = 0
    gifs=True

    generate_instances_figures(evaluate_container,savepath,modelname,instance_list,gifs=gifs,mod_ind=modality_indice,fac_ind=factor_indice)
    generate_instance_performance_figure(evaluate_container,savepath,modelname,show=True)


    # DISPLAY TRIALS 
    model_folder = os.path.join(savepath,modelname)
    for instance in range(Ninstances) :
        for trial in [trials_per_instances-1] :
            full_file_name = ActiveSaveManager.generate_save_name(model_folder,instance,trial,'f')
            trial_plot(full_file_name,title="Trial " + str(trial) + " sum-up (instance " + str(instance) + " )")
    input()

def run_models(savepath,models_dictionnary,Ntrials,Ninstances,overwrite = False,verbose=False):
    max_n = len(models_dictionnary)
    cnter = 0.0
    for key in models_dictionnary: 
        print("MODEL : " + key)
        model_options = models_dictionnary[key]
        run_model(savepath,key,model_options,Ntrials,Ninstances,overwrite=overwrite,global_prop=[cnter,max_n],verbose=verbose)

        cnter = cnter + 1

def run_model(savepath,model_name,model_options,Ntrials,Ninstances,overwrite = False,global_prop=[0,1],verbose=False):
    a_learn = model_options[0]
    a_acc = model_options[1]
    a_str = model_options[2]
    b_learn = model_options[3]
    b_acc = model_options[4]
    b_str = model_options[5]
    d_learn = model_options[6]
    memory_decay_type = model_options[7]
    memory_decay_halftime = model_options[8]

    # SIMULATING TRAINING
    model = nf_model(model_name,savepath,prop_poubelle=0.0,prior_a_ratio=a_acc,prior_a_strength=a_str,learn_a=a_learn,
                                                        prior_b_ratio=b_acc,prior_b_strength=b_str,learn_b=b_learn,
                                                        learn_d=d_learn,
                                                        mem_dec_type=memory_decay_type,mem_dec_halftime=memory_decay_halftime,
                                                        verbose=verbose)
    model.initialize_n_layers(Ninstances)
    trial_times = [0.01]
    model.run_n_trials(Ntrials,overwrite=overwrite,global_prop=global_prop,list_of_last_n_trial_times=trial_times)

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def sliding_window_mean(list_input,window_size = 5):
        list_output = []
        N = len(list_input)
        for trial in range(N):
            mean_value = 0
            counter = 0
            for k in range(trial - window_size,trial + window_size + 1):
                if(k>=0):
                    try :
                        mean_value += list_input[k]
                        counter += 1
                    except :
                        a = 0
                        #Nothing lol
            list_output.append(mean_value/counter)
        return list_output

def generate_a_dictionnary(a_priors,b_priors) :
    new_dict = {}
    for ka in range(a_priors.shape[0]):
        for kb in range(b_priors.shape[0]):
            modelchar = [True,a_priors[ka],1,True,b_priors[kb],1,True,MemoryDecayType.NO_MEMORY_DECAY,2000]
            modelname = "a_ac"+str(int(10*a_priors[ka]))+"_str1_b_ac"+str(int(10*b_priors[kb]))+"_str1"
            new_dict[modelname] = modelchar
    return new_dict








def brouillon():

    save_path = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","code","results","series","series_a_b_prior")
    models_dictionnary = {
        "a_ac1p5_str1_b_ac1_str1":[True,1.5,1,True,1,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000],
        "a_ac3_str1_b_ac1_str1":[True,3,1,True,1,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000],
        "a_ac5_str1_b_ac1_str1":[True,5,1,True,1,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000],
        "a_ac10_str1_b_ac1_str1":[True,10,1,True,1,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000],
        "a_ac15_str1_b_ac1_str1":[True,15,1,True,1,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000],
        "a_ac25_str1_b_ac1_str1":[True,25,1,True,1,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000],
        "a_ac50_str1_b_ac1_str1":[True,50,1,True,1,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000],
        "a_ac200_str1_b_ac1_str1":[True,200,1,True,1,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000]
    }

    prior_value_a = np.array([1.0,1.2,1.5,1.8,2.0,2.4,2.8,3.0,5.0,15.0,50.0,200.0])
    #prior_value_a = np.array([1.0,1.2,1.5,2.0,5.0,15.0])
    models_dictionnary = (generate_a_dictionnary(prior_value_a,prior_value_a))
    Ninstances = 10
    Ntrials = 500
    overwrite = False
    run_models(save_path,models_dictionnary,Ntrials,Ninstances,overwrite=overwrite)

    # Multimodel plot :
    all_beh_err = []
    all_stat_err = []
    all_a_err = []
    all_b_err = []

    # cnt = 0
    # for key in models_dictionnary:
    #     print(key)
    #     mean_A,mean_B,mean_D,a_err_arr,b_err_arr,Ka_arr,Kb_arr,Kd_arr,error_states_arr,error_behaviour_arr,total_instances = mean_indicators_model(key,savepath)
    #     all_beh_err.append(error_behaviour_arr)
    #     all_stat_err.append(error_states_arr)
    #     all_a_err.append(a_err_arr)
    #     all_b_err.append(b_err_arr)

    t = np.arange(0,Ntrials,1)
    # arr  = (np.array(all_stat_err))

    savenam = os.path.join(save_path,"output_array.my_arr")
    # save_flexible(arr,savenam)

    arr = load_flexible(savenam)
    # print(arr.shape)
    # for i in range(len(models_dictionnary)):
    #     y = arr[i,:]
    #     if y.shape[0]>Ntrials:
    #         y = y[:Ntrials]
    #     plt.scatter(t,y,s=1)

    # # Single value of a or b

    # for i in range(len(models_dictionnary)):
    #     y = arr[i,:]
    #     t = np.arange(0,Ntrials,1)
    #     y_av = sliding_window_mean(list(y),4)
    #     y_av = np.array(y_av)
    #     if y_av.shape[0]>Ntrials:
    #         y_av = y_av[:Ntrials]
        
    #     my_key = list(models_dictionnary)[i]
    #     list_of_key =  (my_key.split("_"))
    #     a_acc = float(list_of_key[1].strip("ac"))/10
    #     b_acc = float(list_of_key[4].strip("ac"))/10
    #     print(a_acc,b_acc)
    #     prior_value = models_dictionnary[list(models_dictionnary)[i]][1]
    #     if (a_acc == 1.0) or (a_acc==1.5):
    #         plt.plot(t,y_av,label="Good Prior Biais =  " + str(prior_value) + " b_acc = " + str(b_acc))

    # plt.legend()
    # plt.xlabel("Trials")
    # plt.ylabel("State error w.r.t optimal")
    # plt.title("How does prior influence overall performance")
    # plt.grid(True)
    # plt.show()

    # 3D plot


    the_t = 100
    xs = []
    ys = []
    zs = []

    icnt = 0
    jcnt = 0
    J = prior_value_a.shape[0]
    plot_this = np.zeros((J,J))

    for i in range(len(models_dictionnary)):
        y = arr[i,:]
        t = np.arange(0,Ntrials,1)
        y_av = sliding_window_mean(list(y),10)
        y_av = np.array(y_av)
        if y_av.shape[0]>Ntrials:
            y_av = y_av[:Ntrials]
        
        my_key = list(models_dictionnary)[i]
        list_of_key =  (my_key.split("_"))
        a_acc = float(list_of_key[1].strip("ac"))/10
        b_acc = float(list_of_key[4].strip("ac"))/10
        #if (a_acc < 50)and(b_acc<50):
        xs.append(a_acc)
        ys.append(b_acc)
        zs.append(y_av[the_t])
        plot_this[icnt,jcnt] = y_av[the_t]

        icnt = icnt+1
        if (icnt>=J):
            icnt = 0
            jcnt = jcnt+1



    from matplotlib import cm
    print(plot_this)
    zs = np.array(zs)
    X,Y = np.meshgrid(prior_value_a,prior_value_a)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs, cmap='Greens')
    ax.plot_surface(X,Y,plot_this,linewidth=0,cmap=cm.coolwarm, antialiased=False)
    plt.show()