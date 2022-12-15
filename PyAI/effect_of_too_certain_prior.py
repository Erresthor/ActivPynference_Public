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
from pyai.models_neurofeedback.article_1_simulations.climb_stairs_flat_priors import nf_model,nf_model_imp5,nf_model_imp4,evaluate_container,nf_model_imp6
from pyai.neurofeedback_run import save_model_performance_dictionnary,load_model_performance_dictionnary,evaluate_model_figure,trial_plot_from_name

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

def figure_from_dico(full_dico):
    smooth_window = 5
    error_observations = (full_dico["mean"]["observation_error"])
    error_states = (full_dico["mean"]["state_error"])
    error_behaviour = (full_dico["mean"]["behaviour_error"])
    a_err = (full_dico["mean"]["a_error"])
    print(a_err)
    b_err = (full_dico["mean"]["b_error"])
    a_unc = (full_dico["mean"]["a_entropy"])
    b_unc = (full_dico["mean"]["b_entropy"])

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

    color1 = 'red'
    color2 = 'blue'
    fig_main = plt.figure()
        
    fig = fig_main
    ax1 = fig.add_subplot(211)
    #ax1.grid(True, which='both')
    
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.spines['bottom'].set_position(('data',0))
    ax1.yaxis.set_ticks_position('left')
    ax1.spines['left'].set_position(('data',0))

    # ax1.set_xticks([0,25,50,75,100,125,150])
    # ax1.set_yticks([1,0.75,0.5,0.25,0])
    gridcolor = np.array([0.5,0.5,0.5,0.2])
    ax1.grid(color=gridcolor, linestyle='-',alpha = 0.2)
    ax1.tick_params(axis='y', labelcolor="black")

    #ax1.set_xlabel('Trials',fontsize=10)
    ax1.set_ylabel('MODEL ERROR', color="black",fontsize=10)
    #ax1.set_ylim([-0.1,1.1])
    

    # ax2 = ax1.twinx()
    # ax2.tick_params(axis='y', labelcolor="black")
    # ax2.set_ylabel('MODEL ERROR (kl divergence)', color="black",fontsize=10)  # we already handled the x-label with ax1
    # ax2.set_ylim([-0.1,1.1])
    wdt = 2.5
    #l1 = ax1.plot(trials, a_unc, color=color1,label='A entropy',ls='--',linewidth=wdt/2.0)
    #l2 = ax1.plot(trials, b_unc, color=color2,label='B entropy',ls='--',linewidth=wdt/2.0)
    # instantiate a second axes that shares the same x-axis
    
    l3 = ax1.plot(trials, a_err, color=color1,label='Perception model error',ls='-',linewidth=wdt)
    l4 = ax1.plot(trials, b_err, color=color2,label='Action model error',ls='-',linewidth=wdt)

    ls = l3 + l4
    labs = [l.get_label() for l in ls]
    ax1.legend(ls,labs,loc = 0)

    
    

    # -----------------------------------------------------------------
    color3 = np.array([1,224.0/255,179.0/255,0.5])
    color3l = 'orange'
    color4 = np.array([1,102.0/255.0,102.0/255.0,0.5])
    color4l = 'red'
    color5 = np.array([0.25,1.0,0.5,0.5])
    color5l = 'green'
    
    ax3 = fig.add_subplot(212,sharex=ax1)
    # ax3.axvline(0,color="black")
    # ax3.axhline(0,color="black")
    #ax3.grid()
    ax3.spines['right'].set_color('none')
    ax3.spines['top'].set_color('none')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.spines['bottom'].set_position(('data',0))
    ax3.yaxis.set_ticks_position('left')
    ax3.spines['left'].set_position(('data',0))

    ax3.set_xticks([0,25,50,75,100,125,150])
    ax3.set_yticks([1,0.75,0.5,0.25,0])
    ax3.grid(color=gridcolor, linestyle='-',alpha = 0.2)

    l1 = ax3.plot(trials,error_states,'+',color=color3)
    #l2 = ax3.plot(trials,error_behaviour,'+',color=color4)
    l3 = ax3.plot(trials,error_observations,"+",color=color5)
    
    l4 = ax3.plot(trials,state_error_mean,"-",color=color3l,label = 'Subject mental state errror (smoothed)',linewidth=wdt)
    
    #l6 = ax3.plot(trials,behaviour_error_mean,"-",color=color4l,label = 'Subject behaviour error (smoothed)',linewidth=wdt)

    l5 = ax3.plot(trials,error_obs_mean,"-",color=color5l,label = 'Feedback error (smoothed)',linewidth=wdt)

    ls = l5 # + l5 #+l6
    labs = [l.get_label() for l in ls]
    ax3.legend(ls,labs,loc = 'best')

    ax3.set_xlabel('Trials',fontsize=15)
    ax3.set_ylabel('FEEDBACK ERROR', color="Black",fontsize=10)
    ax3.tick_params(axis='y', labelcolor='black',size= 15)
    ax3.set_ylim([0.0,1.0])
    #ax3.set_ylim(ax1.get_ylim()[::-1])

    fig_main.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.suptitle("Trial sumup", fontsize=16,y=1.08)

    return fig


if __name__=="__main__":
    save_path = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","TEMPORARY_TEST_BED","article_plots","wtf_is_going_on--2")
    overwrite = True
    Ntrials = 50

    b_val = 1000

    lambdas = [0,0.1,1,4,9,99]
    dicos = []
    dico12 = [[],[]]
    for lambda_value in lambdas:
        model_name = "feedback = "+str(lambda_value)+"_"+str(b_val)

        # strength_a = 1
        # strength_b = 1
        # b_prior = b_val
        # a_prior = lambda_value + 1
        # model = nf_model_imp4(model_name,save_path,prop_poubelle = 0.3,
        #         learn_a = True,prior_a_ratio = a_prior,prior_a_strength=strength_a,
        #         learn_b= True,prior_b_ratio = b_prior,prior_b_strength=strength_b,
        #         learn_d= True,
        #         perfect_a = False,perfect_b=False,verbose = False,SHAM="False")
        # model.initialize_n_layers(10)
        # model.run_n_trials(Ntrials)
        # complete_data = True
        # var = True 
        # save_model_performance_dictionnary(save_path,model_name,evaluate_container,overwrite=overwrite,include_var=var,include_complete=complete_data)
        # full_dico = load_model_performance_dictionnary(save_path,model_name,var,complete_data)
        # dicos.append(full_dico)



        model_name = "feedback = "+str(lambda_value)+"_"+str(b_val)+"noisy1"
        strength_a = 0.1
        strength_b = 1
        b_prior = b_val
        a_prior = lambda_value + 1
        model = nf_model_imp6(model_name,save_path,prop_poubelle = 0.3,
                learn_a = True,prior_a_ratio = a_prior,prior_a_strength=strength_a,
                learn_b = True,prior_b_ratio = b_prior,prior_b_strength=strength_b,
                learn_d = True,
                perfect_a = False,perfect_b=False,verbose = False,SHAM="False",noiselevel=1)
        model.initialize_n_layers(10)
        model.run_n_trials(Ntrials)
        complete_data = True
        var = True 
        save_model_performance_dictionnary(save_path,model_name,evaluate_container,overwrite=overwrite,include_var=var,include_complete=complete_data)
        full_dico = load_model_performance_dictionnary(save_path,model_name,var,complete_data)
        dico12[0].append(full_dico)

        # trial_plot_from_name(save_path,model_name,7,[0,1,2,3,4,5],old=True)


        model_name = "feedback = "+str(lambda_value)+"_"+str(b_val)+"noisy2"
        strength_a = 0.1
        strength_b = 1
        b_prior = b_val
        a_prior = lambda_value + 1
        model = nf_model_imp6(model_name,save_path,prop_poubelle = 0.3,
                learn_a = True,prior_a_ratio = a_prior,prior_a_strength=strength_a,
                learn_b = True,prior_b_ratio = b_prior,prior_b_strength=strength_b,
                learn_d = True,
                perfect_a = False,perfect_b=False,verbose = False,SHAM="False",noiselevel=2)
        model.initialize_n_layers(10)
        model.run_n_trials(Ntrials,overwrite=False)
        complete_data = True
        var = True 
        save_model_performance_dictionnary(save_path,model_name,evaluate_container,overwrite=overwrite,include_var=var,include_complete=complete_data)
        full_dico = load_model_performance_dictionnary(save_path,model_name,var,complete_data)
        dico12[1].append(full_dico)

        # trial_plot_from_name(save_path,model_name,7,[0,1,2,3,4,5])
        # input()
        # plt.close()


    # complete_data = True
    # var = True 
    # save_model_performance_dictionnary(save_path,model_name,evaluate_container,overwrite=overwrite,include_var=var,include_complete=complete_data)
    # full_dico = load_model_performance_dictionnary(save_path,model_name,var,complete_data)
    # dicos.append(full_dico)
    # save_folder = os.path.join(save_path,model_name,"_RESULTS_" + "ok")
    # if not os.path.exists(save_folder):
    #             try:
    #                 os.makedirs(save_folder)
    #             except OSError as exc: # Guard against race condition
    #                 raise
    

    # for dic in dicos:
    #     fig = figure_from_dico(dic)
    simu = ["close","far"]
    i = 0
    
    for little_dic in dico12 :
        stri = simu[i]
        j=0
        for dic in little_dic :
            fig = figure_from_dico(dic)
            fig.suptitle(stri+str(lambdas[j]))
            j=j+1
        i = i + 1
    # figname = os.path.join(save_folder,"performances")
    # plt.savefig(figname,bbox_inches='tight',dpi=1000)
    if(True):
        plt.draw()
    else :
        plt.close()
    plt.show()