import sys,os

from cmath import pi
from turtle import width
from imageio import save
import numpy as np
from pyai.base.miscellaneous_toolbox import smooth_1D_array
from pyai.base.matrix_functions import *
from pyai.base.function_toolbox import spm_wnorm
from pyai.neurofeedback_run import save_model_performance_dictionnary,load_model_performance_dictionnary
import matplotlib.pyplot as plt
from pyai.models_neurofeedback.article_1_simulations.climb_stairs_flat_priors import nf_model,nf_model_imp2,nf_model_imp,evaluate_container

def sliding_window_mean(array_input,window_size = 5):
        list_output = np.zeros(array_input.shape)
        N = array_input.shape[0]
        for trial in range(N):
            mean_value = 0
            counter = 0
            for k in range(trial - window_size,trial + window_size + 1):
                if(k>=0):
                    try :
                        mean_value += array_input[k]
                        counter += 1
                    except :
                        a = 0
                        #Nothing lol
            list_output[trial] = mean_value/counter
        return list_output

if __name__=="__main__":
    save_path = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","TEMPORARY_TEST_BED","one_more_trial")
    overwrite = True
    Ntrials = 150
    model_name = "learn_b_a_known_low_conf_3"

    model = nf_model(model_name,save_path,prop_poubelle=0.5,
                        perfect_a=True,learn_a= False,
                        perfect_b=False,learn_b=True,prior_b_ratio=1,prior_b_strength=0.1,
                        learn_d=True,verbose=True)
    model.initialize_n_layers(10)
    model.run_n_trials(Ntrials)
    complete_data = True
    var = True 
    save_model_performance_dictionnary(save_path,model_name,evaluate_container,overwrite=overwrite,include_var=var,include_complete=complete_data)
    full_dico = load_model_performance_dictionnary(save_path,model_name,var,complete_data)

    # # OK 
    modelok = nf_model_imp(model_name+"_ok",save_path,prop_poubelle=0.5,
                        perfect_a=True,learn_a= False,
                        perfect_b=False,learn_b=True,prior_b_ratio=1,prior_b_strength=0.1,
                        learn_d=True,verbose=True,SHAM="False")
    modelok.initialize_n_layers(10)
    modelok.run_n_trials(Ntrials)
    complete_data = True
    var = True 
    save_model_performance_dictionnary(save_path,model_name+"_ok",evaluate_container,overwrite=overwrite,include_var=var,include_complete=complete_data)
    full_dico_ok = load_model_performance_dictionnary(save_path,model_name+"_ok",var,complete_data)



    # IMPERFECT
    overwrite=False
    modelimp = nf_model_imp2(model_name+"_imp",save_path,prop_poubelle=0.5,
                        perfect_a=True,learn_a= False,
                        perfect_b=False,learn_b=True,prior_b_ratio=1,prior_b_strength=0.1,
                        learn_d=True,verbose=True,SHAM="False")
    modelimp.initialize_n_layers(10)
    modelimp.run_n_trials(Ntrials,overwrite=overwrite)
    complete_data = True
    var = True 
    save_model_performance_dictionnary(save_path,model_name+"_imp",evaluate_container,overwrite=overwrite,include_var=var,include_complete=complete_data)
    full_dico_imp = load_model_performance_dictionnary(save_path,model_name+"_imp",var,complete_data)


    # SHAM
    modelsham = nf_model(model_name+"_SHAM",save_path,prop_poubelle=0.5,
                        perfect_a=True,learn_a= False,
                        perfect_b=False,learn_b=True,prior_b_ratio=1,prior_b_strength=0.1,
                        learn_d=True,verbose=True,SHAM="True")
    modelsham.initialize_n_layers(10)
    modelsham.run_n_trials(Ntrials)
    complete_data = True
    var = True 
    save_model_performance_dictionnary(save_path,model_name+"_SHAM",evaluate_container,overwrite=overwrite,include_var=var,include_complete=complete_data)
    full_dico_sham = load_model_performance_dictionnary(save_path,model_name+"_SHAM",var,complete_data)


    # model_name = "learn_b_a_known_no_poub_low_conf_learn_a_veryhigh_conf"
    # model = nf_model(model_name,save_path,prop_poubelle=0.0,
    #                     perfect_a=False,learn_a=True,prior_a_ratio=5,prior_a_strength=100,
    #                     perfect_b=False,learn_b=True,prior_b_ratio=1,prior_b_strength=0.1,
    #                     learn_d=True,verbose=True)
    # model.initialize_n_layers(10)
    # model.run_n_trials(Ntrials)
    # complete_data = True
    # var = True 
    # save_model_performance_dictionnary(save_path,model_name,evaluate_container,overwrite=overwrite,include_var=var,include_complete=complete_data)
    # full_dico = load_model_performance_dictionnary(save_path,model_name,var,complete_data)

    smooth_window = 5
    error_observations = (full_dico["mean"]["observation_error"])
    error_states = (full_dico["mean"]["state_error"])
    error_behaviour = (full_dico["mean"]["behaviour_error"])
    a_err = (full_dico["mean"]["a_error"])
    b_err = (full_dico["mean"]["b_error"])
    a_unc = (full_dico["mean"]["a_entropy"])
    b_unc = (full_dico["mean"]["b_entropy"])


    error_observations_sham = (full_dico_sham["mean"]["observation_error"])
    error_states_sham = (full_dico_sham["mean"]["state_error"])
    error_behaviour_sham = (full_dico_sham["mean"]["behaviour_error"])
    a_err_sham = (full_dico_sham["mean"]["a_error"])
    b_err_sham = (full_dico_sham["mean"]["b_error"])
    a_unc_sham = (full_dico_sham["mean"]["a_entropy"])
    b_unc_sham = (full_dico_sham["mean"]["b_entropy"])


    error_observations_ok = (full_dico_ok["mean"]["observation_error"])
    error_states_ok = (full_dico_ok["mean"]["state_error"])
    error_behaviour_ok = (full_dico_ok["mean"]["behaviour_error"])
    a_err_ok = (full_dico_ok["mean"]["a_error"])
    b_err_ok = (full_dico_ok["mean"]["b_error"])
    a_unc_ok = (full_dico_ok["mean"]["a_entropy"])
    b_unc_ok = (full_dico_ok["mean"]["b_entropy"])


    error_observations_imp = (full_dico_imp["mean"]["observation_error"])
    error_states_imp = (full_dico_imp["mean"]["state_error"])
    error_behaviour_imp = (full_dico_imp["mean"]["behaviour_error"])
    a_err_imp = (full_dico_imp["mean"]["a_error"])
    b_err_imp = (full_dico_imp["mean"]["b_error"])
    a_unc_imp = (full_dico_imp["mean"]["a_entropy"])
    b_unc_imp = (full_dico_imp["mean"]["b_entropy"])

    print(error_states_sham)
    print(error_observations_sham)
    from pyai.model.model_visualizer import general_performance_plot

    def sliding_window_mean(list_input,window_size = 3):
        list_output = []
        N = len(list_input)
        for trial in range(N):
            
            mean_value = 0
            counter = 0
            for k in range(trial - window_size,trial + window_size + 1):
                if(k>=0):
                    try :
                        mean_value += list_input[k][0] #First factor
                        counter += 1
                    except :
                        a = 0
                        #Nothing lol
            if (counter==0) :
                list_output.append(0)
            else : 
                list_output.append(mean_value/counter)
        return list_output

    trials = np.linspace(0,Ntrials,Ntrials)
    state_error_mean = sliding_window_mean(error_states,smooth_window)
    behaviour_error_mean = sliding_window_mean(error_behaviour,smooth_window)
    error_obs_mean = sliding_window_mean(error_observations,smooth_window)

    state_error_mean_ok = sliding_window_mean(error_states_ok,smooth_window)
    behaviour_error_mean_ok = sliding_window_mean(error_behaviour_ok,smooth_window)
    error_obs_mean_ok = sliding_window_mean(error_observations_ok,smooth_window)

    state_error_mean_imp = sliding_window_mean(error_states_imp,smooth_window)
    behaviour_error_mean_imp = sliding_window_mean(error_behaviour_imp,smooth_window)
    error_obs_mean_imp = sliding_window_mean(error_observations_imp,smooth_window)

    state_error_mean_sham = sliding_window_mean(error_states_sham,smooth_window)
    behaviour_error_mean_sham = sliding_window_mean(error_behaviour_sham,smooth_window)
    error_obs_mean_sham = sliding_window_mean(error_observations_sham,smooth_window)

    color1 = 'red'
    color2 = 'blue'
    color3 = 'green'
    color4 = 'orange'
    fig_main = plt.figure()
    ax1 = fig_main.add_subplot(111)
    #ax1.grid(True, which='both')
    
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.spines['bottom'].set_position(('data',0))
    ax1.yaxis.set_ticks_position('left')
    ax1.spines['left'].set_position(('data',0))

    ax1.set_xticks([0,25,50,75,100,125,150])
    ax1.set_yticks([1,0.75,0.5,0.25,0])
    gridcolor = np.array([0.5,0.5,0.5,0.2])
    ax1.grid(color=gridcolor, linestyle='-',alpha = 0.2)
    ax1.tick_params(axis='y', labelcolor="black")

    #ax1.set_xlabel('Trials',fontsize=10)
    ax1.set_ylabel('ACTION MODEL ERROR', color="black",fontsize=10)
    ax1.set_ylim([-0.1,1.1])
    

    # ax2 = ax1.twinx()
    # ax2.tick_params(axis='y', labelcolor="black")
    # ax2.set_ylabel('MODEL ERROR (kl divergence)', color="black",fontsize=10)  # we already handled the x-label with ax1
    # ax2.set_ylim([-0.1,1.1])
    wdt = 2.5
    # instantiate a second axes that shares the same x-axis
    
    l1 = ax1.plot(trials, b_err, color=color1,label='PERFECT',ls='-',linewidth=wdt)
    l2 = ax1.plot(trials, b_err_ok, color=color2,label='NOISY',ls='-',linewidth=wdt)
    # l3 = ax1.plot(trials, b_err_imp, color=color3,label='VERY NOISY',ls='-',linewidth=wdt)
    l4 = ax1.plot(trials, b_err_sham, color=color4,label='SHAM',ls='-',linewidth=wdt)

    # ls = l1 + l2 + l3 + l4
    ls = l1 + l2 + l4
    labs = [l.get_label() for l in ls]
    ax1.legend(ls,labs,loc = 0)

    
    
    fig_main = plt.figure()

    figs=  fig_main.subfigures(1,2)
    
    fig = figs[0]
    # -----------------------------------------------------------------
    color1l = np.array([1,102.0/255.0,102.0/255.0,0.5])
    color2l = np.array([102.0/255.0,102.0/255.0,1,0.5])
    color3l = np.array([0.25,1.0,0.5,0.5])
    color4l = np.array([1,224.0/255,179.0/255,0.5])
    ax2 = fig.add_subplot(211,sharex=ax1)

    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.spines['bottom'].set_position(('data',0))
    ax2.yaxis.set_ticks_position('left')
    ax2.spines['left'].set_position(('data',0))
    ax2.set_ylabel('STATE ERROR / OPTIMAL', color="Black",fontsize=10)

    ax2.set_xticks([0,25,50,75,100,125,150])
    ax2.set_yticks([1,0.75,0.5,0.25,0])
    ax2.grid(color=gridcolor, linestyle='-',alpha = 0.2)

    l1 = ax2.plot(trials,error_states,'+',color=color1l)
    l2 = ax2.plot(trials,error_states_ok,'+',color=color2l)
    #l3 = ax2.plot(trials,error_states_imp,'+',color=color3l)
    l4 = ax2.plot(trials,error_states_sham,'+',color=color4l)

    l5 = ax2.plot(trials,state_error_mean,'-',color=color1)
    l6 = ax2.plot(trials,state_error_mean_ok,'-',color=color2)
    #l7 = ax2.plot(trials,state_error_mean_imp,'-',color=color3)
    l8 = ax2.plot(trials,state_error_mean_sham,'-',color=color4)

    ax2.set_ylim([-0.1,1.1])


    ax3 = fig.add_subplot(212,sharex=ax2)
    # ax3.axvline(0,color="black")
    # ax3.axhline(0,color="black")
    #ax3.grid()
    ax3.spines['right'].set_color('none')
    ax3.spines['top'].set_color('none')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.spines['bottom'].set_position(('data',0))
    ax3.yaxis.set_ticks_position('left')
    ax3.spines['left'].set_position(('data',0))

    l10 = ax3.plot(trials,error_observations,'x',color=color1l)
    l20 = ax3.plot(trials,error_observations_ok,'x',color=color2l)
    #l30 = ax3.plot(trials,error_observations_imp,'x',color=color3l)
    l40 = ax3.plot(trials,error_observations_sham,'x',color=color4l)

    l50 = ax3.plot(trials,error_obs_mean,'--',color=color1)
    l60 = ax3.plot(trials,error_obs_mean_ok,'--',color=color2)
    #l70 = ax3.plot(trials,error_obs_mean_imp,'--',color=color3)
    l80 = ax3.plot(trials,error_obs_mean_sham,'--',color=color4)

    ls = l4 + l5 +l6
    labs = [l.get_label() for l in ls]
    ax3.legend(ls,labs,loc = 'best')

    ax3.set_xlabel('Trials',fontsize=15)
    ax3.set_ylabel('FEEDBACK ERROR / OPTIMAL', color="Black",fontsize=10)
    ax3.tick_params(axis='y', labelcolor='black',size= 15)
    ax3.set_ylim([0,1])
    #ax3.set_ylim(ax1.get_ylim()[::-1])

    fig_main.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.suptitle("Trial sumup", fontsize=16,y=1.08)

    save_folder = os.path.join(save_path,model_name,"_RESULTS_" + "ok")
    
    
    fig_2 = figs[1]
    colors = [color1,color2,color3]
    N = 100
    ax = fig_2.add_subplot(211)
    ax.grid(zorder=0)
    sop_std = np.average(np.sqrt(full_dico["variance"]["state_error"][N:,:]))
    soo_std = np.average(np.sqrt(full_dico_ok["variance"]["state_error"][N:,:]))
    #soi_std = np.average(np.sqrt(full_dico_imp["variance"]["state_error"][N:,:]))
    sos_std = np.average(np.sqrt(full_dico_sham["variance"]["state_error"][N:,:]))
    # so_err = [sop_std,soo_std,soi_std,sos_std]
    so_err = [sop_std,soo_std,sos_std]
    # ax.bar(["Perfect",'Noisy',"Very noisy","SHAM"],[np.average(error_states[N:,:]),np.average(error_states_ok[N:,:]),np.average(error_states_imp[N:,:]),np.average(error_states_sham[N:,:])],
    #             yerr=so_err,color=colors,
    #             zorder=3)
    ax.bar(["Perfect",'Noisy',"SHAM"],[np.average(error_states[N:,:]),np.average(error_states_ok[N:,:]),
                np.average(error_states_sham[N:,:])],
                yerr=so_err,color=colors,
                zorder=3)
    ax.set_ylim([0,1])
    ax.set_xlabel("Feedback quality")
    ax.set_ylabel("Hidden state error (over the 50 last trials)")
    
    ax = fig_2.add_subplot(212)
    ax.grid(zorder=0)
    eop_std = np.average(np.sqrt(full_dico["variance"]["observation_error"][N:,:]))
    eoo_std = np.average(np.sqrt(full_dico_ok["variance"]["observation_error"][N:,:]))
    # eoi_std = np.average(np.sqrt(full_dico_imp["variance"]["observation_error"][N:,:]))
    eos_std = np.average(np.sqrt(full_dico_sham["variance"]["observation_error"][N:,:]))
    # eo_err = [eop_std,eoo_std,eoi_std,eos_std]
    eo_err = [eop_std,eoo_std,eos_std]

    # ax.bar(["Perfect",'Noisy',"Very noisy","SHAM"],[np.average(error_observations[N:,:]),np.average(error_observations_ok[N:,:]),np.average(error_observations_imp[N:,:]),np.average(error_observations_sham[N:,:])],
    #             yerr=eo_err,color=colors,
    #             zorder=3)
    
    ax.bar(["Perfect",'Noisy',"SHAM"],[np.average(error_observations[N:,:]),np.average(error_observations_ok[N:,:]),
                np.average(error_observations_sham[N:,:])],
                yerr=eo_err,color=colors,
                zorder=3)
    ax.set_ylim([0,1])
    ax.set_xlabel("Feedback quality")
    ax.set_ylabel("Feedback error (over the 50 last trials)")
    
    
    if not os.path.exists(save_folder):
                try:
                    os.makedirs(save_folder)
                except OSError as exc: # Guard against race condition
                    raise
    
    figname = os.path.join(save_folder,"performances")
    plt.savefig(figname,bbox_inches='tight',dpi=1000)
    if(True):
        plt.draw()
    else :
        plt.close()
    plt.show()