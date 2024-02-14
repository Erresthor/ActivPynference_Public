import numpy as np
import statistics as stat
import scipy.stats as scistats
import math,sys,os
import pickle
import matplotlib.pyplot as plt
import scipy.optimize as opt
from matplotlib.animation import FuncAnimation, PillowWriter

import actynf
from tools import clever_running_mean,color_spectrum,clever_running_mean_mess,dist_kl_dir

from m1_model import neurofeedback_training

# Useful function, might add it to the network class !
def simulate_and_save(my_net,savepath,Nsubj,Ntrials,override=False):
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))

    exists = os.path.isfile(savepath)
    if (not(exists)) or (override):
        stm_subjs = []
        weight_subjs = []
        print("Saving to " + savepath)
        for sub in range(Nsubj):
            subj_net = my_net.copy_network(sub)

            STMs,weights = subj_net.run_N_trials(Ntrials,return_STMs=True,return_weights=True)
            stm_subjs.append(STMs)
            weight_subjs.append(weights)

        save_this = {
            "stms": stm_subjs,
            "matrices" : weight_subjs
        }

        with open(savepath, 'wb') as handle:
            pickle.dump(save_this, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved to :   " + savepath)

def extract_training_data(savepath):
    # EXTRACT TRAINING CURVES
    with open(savepath, 'rb') as handle:
        saved_data = pickle.load(handle)
    stms = saved_data["stms"]
    weights = saved_data["matrices"]

    Nsubj = len(stms)
    Ntrials = len(weights[0])-1 # One off because we save the initial weights (= trial 0)
    return stms,weights,Nsubj,Ntrials

def exploit_saved_data_old(savepathlist,namelist):
    trialsLastN = 20
    plot_idx = 3
    pointcloud_plot = True

    if pointcloud_plot:
        # Pointcloud plots
        fig,axes = plt.subplots(1,len(savepathlist),sharey=True)
        for k,savepath in enumerate(savepathlist):
            ax = axes[k]
            ax.grid()
            ax.set_title(namelist[k])
            ax.set_ylim([0.0,4.0])
            onlyfiles = [os.path.join(savepath, f) for f in os.listdir(savepath) if os.path.isfile(os.path.join(savepath, f))]

            xs = []
            ys = []
            for file in onlyfiles:
                param_val = file.split(".")
                true_fb_std = float(param_val[-3] + '.' + param_val[-2])

                _stm,_weight,_Nsubj,_Ntrials = extract_training_data(file)

                state_perfs = [_stm[0][idx][0].x[0] for idx in range(1,Ntrials+1)]
                state_perfs_array = np.array(state_perfs)

                xs.append(true_fb_std)
                ys.append(np.mean(state_perfs_array[-trialsLastN:,:]))

                if plot_idx == 2:
                    # Plot the training curves of every subject with a color
                    # depending on the true_b_std:
                    t = true_fb_std/2.50
                    color = np.array([1.0,0.0,0.0,0.2])*t + np.array([0.0,0.0,1.0,0.2])*(1-t)
                    ax.plot(np.linspace(0,Ntrials,Ntrials),np.mean(state_perfs_array,axis=-1),color=color)

            ax.scatter(xs,ys,color='blue',s=1)

            if (plot_idx==1):
                # FITTING
                # Plot the final alignment of NF subjects

                # def f(x, a, b, c):
                #     """ Logistic function """
                #     return a/(1+b*np.exp(c*x))

                def f(x, a, b, c,d):
                    """ Polynomial degree 3 function """
                    return a*np.power(x,3) + b*np.power(x,2) + c*np.power(x,1) + d

                # a, c = np.random.exponential(size=2)
                # b, d = np.random.randn(2)

                popt,pcov = opt.curve_fit(f, xs, ys)
                print(popt)
                (a_, b_, c_,d_) = popt
                
                # theta_0 = [1.0,2.0,1.0]
                # popt, pcov = opt.leastsq(residual, theta_0, args=(xs,ys))
                # print(popt)
                x_fit = np.linspace(0,2.5,1000)
                y_fit = f(x_fit,a_,b_,c_,d_)
                ax.plot(x_fit, y_fit,color='red',linewidth=2.0)

            if plot_idx==3:
                redcol = np.array([1.0,0.0,0.0])
                fullcol = np.concatenate([redcol,np.array([1.0])])
                transcol = np.concatenate([redcol,np.array([0.2])])
                X,Y,varY = clever_running_mean_mess(xs,ys,30)
                ax.fill_between(X,Y-varY,Y+varY,color=transcol)
                ax.plot(X,Y,color=fullcol)


        fig.show()
        input()

    exit()
    # Common plots
    fig,ax = plt.subplots(1)
    for k,savepath in enumerate(savepathlist):
        redcol = np.array([1.0,0.0,0.0])
        redcol = np.random.random((3,))

        ax.grid()
        ax.set_title(namelist[k])
        ax.set_ylim([0.0,4.0])
        onlyfiles = [os.path.join(savepath, f) for f in os.listdir(savepath) if os.path.isfile(os.path.join(savepath, f))]

        xs = []
        ys = []
        for file in onlyfiles:
            param_val = file.split(".")
            true_fb_std = float(param_val[-3] + '.' + param_val[-2])

            _stm,_weight,_Nsubj,_Ntrials = extract_training_data(file)

            state_perfs = [_stm[0][idx][0].x[0] for idx in range(1,Ntrials+1)]
            state_perfs_array = np.array(state_perfs)

            xs.append(true_fb_std)
            ys.append(np.mean(state_perfs_array[-trialsLastN:,:]))

        fullcol = np.concatenate([redcol,np.array([1.0])])
        transcol = np.concatenate([redcol,np.array([0.2])])
        X,Y,varY = clever_running_mean_mess(xs,ys,0.25,1000)
        ax.fill_between(X,Y-varY,Y+varY,color=transcol)
        ax.plot(X,Y,color=fullcol)


    fig.show()
    input()

def exploit_saved_data(savepathlist,trustlevellist):
    trialsLastN = 20
    
    # Common plots
    fig,axes = plt.subplots(1,2)
    fig.suptitle("Cognitive regulation depending on feedback quality for various subject trust levels")
    ax  = axes[0]
    ax_act = axes[1]
    ax.grid()
    ax_act.grid()
    
    ax.set_ylim([0.0,4.0])
    # ax_act.set_ylim(bottom=0.0)

    ax.set_xlabel("Feedback noise")
    ax_act.set_xlabel("Feedback noise")

    ax.set_ylabel("Subject average cognitive state (last " + str(trialsLastN) + " trials)")
    ax_act.set_ylabel("End-of-training subject model quality")

    for k,savepath in enumerate(savepathlist):
        onlyfiles = [os.path.join(savepath, f) for f in os.listdir(savepath) if os.path.isfile(os.path.join(savepath, f))]

        xs = []
        ys = []
        zs = []
        for file in onlyfiles:
            param_val = file.split(".")
            true_fb_std = float(param_val[-3] + '.' + param_val[-2])

            _stm,_weight,_Nsubj,_Ntrials = extract_training_data(file)

            state_perfs = [_stm[0][idx][0].x[0] for idx in range(1,Ntrials+1)]
            action_model = [dist_kl_dir(_weight[0][idx][1]["b"][0],_weight[0][idx][0]["b"][0]) for idx in range(1,Ntrials+1)]
            state_perfs_array = np.array(state_perfs)
            action_model_array = np.array(action_model)
            
            xs.append(true_fb_std)
            ys.append(np.mean(state_perfs_array[-trialsLastN:,:]))
            zs.append(action_model_array[-1])
        
        X,Y,vY = clever_running_mean_mess(xs,ys,30)
        X,Z,vZ = clever_running_mean_mess(xs,zs,30)
        # ax.fill_between(X,Y-varY,Y+varY,color=transcol)
                
        
        ax.fill_between(X,Y-vY,Y+vY,alpha=0.2)
        ax.scatter(xs,ys,s=1)
        ax.plot(X,Y,label=trustlevellist[k])
        
        ax_act.fill_between(X,Z-vZ,Z+vZ,alpha=0.2)
        ax_act.scatter(xs,zs,s=1)
        ax_act.plot(X,Z,label=trustlevellist[k])
        
        # ax.plot(X,Y,color=fullcol)

    ax.legend()
    ax_act.legend()
    fig.show()
    input()


if __name__ == "__main__":
    N = 50

    Nsubj_per_point = 1

    # MODEL PARAMETERS
    Ntrials = 100
    action_selection_inverse_temp = 32.0
    belief_feedback_std = 0.01

    learn_a = False
    clamp_gaussian = False

    T = 10
    Th = 2
    feedback_resolution = 5

    subj_cognitive_resolution = 5
    true_cognitive_resolution = 5

    k1b = 0.01
    epsilon_b = 0.01

    k1a = 10.0 # 5.0 / 1.0 / None
    epsilon_a = 1.0/101.0 # a0 = norm((1/101)* ones + gaussian_prior)*k1a

    k1d = 1.0
    epsilon_d = 1.0

    neutral_action_prop = 0.2 # 20% of the actions have no interest for the task

    pLow = 0.5   # Without any increasing action, there is a pLow chance that the cognitive state will decrease spontaneously
    pUp  = 0.99

    Nsubj = 10


    def savepath_func(belief_fb_std,true_fb_std,action_selection_inverse_temp,learn_a_bool):
        learn_a_rule = (("learn_a_"+str(k1a)) if learn_a_bool else "no_learn_a")
        return os.path.join("simulation_outputs","paper1","stairs","RANDOM_FB",learn_a_rule,str(action_selection_inverse_temp),"subject_expects_feedback_std_"+str(belief_fb_std)+"_noClamp","simulations_3."+str(true_fb_std)+".pickle")

    # for belief_fb_std in belief_feedback_std:
    #     for true_fb_std in true_feedback_std:

    plot_those = np.random.uniform(0.01,2.50,(N,))

    DIR = os.path.dirname(savepath_func(belief_feedback_std,0,action_selection_inverse_temp,learn_a))
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    k = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    while k<N:
        print()
        print("POINT " + str(k+1) + " / " + str(N))
        print()

        true_fb_val = np.random.uniform(0.01,2.50)
        net = neurofeedback_training(T,Th,  # Trial duration + temporal horizon
            subj_cognitive_resolution,true_cognitive_resolution,       # Subject belief about cognitive resolution / true cognitive resolution
            feedback_resolution,feedback_resolution,       # Subject belief about feedback resolution / true feedback resolution
            belief_feedback_std,true_fb_val,   # Subject belief about feedback noise / true feedback noise
            neutral_action_prop,       # how many actions have no impact on the state ?
            k1b,epsilon_b,  # Action mapping previous knowledge
            k1a,epsilon_a,   # Feedback mapping previous knowledge
            k1d,epsilon_d,   # d mapping previous knowledge
            pLow,pUp,   # How likely it is that the cognitive state will go down when unattended
                        # / how likely it is that the correct action will increase the cognitive state
            clamp_gaussian=clamp_gaussian,asit = action_selection_inverse_temp,
            learn_a=learn_a)
                        # Clamp : Weither to increase the categorical probabilistic weights
                        # on the edges or not
                        # asit : inverse temperature of the action selection process
                        # learn_a : Weither to learn the perception matrix on the go

        savepath = savepath_func(belief_feedback_std,true_fb_val,action_selection_inverse_temp,learn_a)
        simulate_and_save(net,savepath,Nsubj_per_point,Ntrials,override=False)

        DIR = os.path.dirname(savepath_func(belief_feedback_std,0,action_selection_inverse_temp,learn_a))
        k = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])



    def plot_4_initial_confidence():
        savepaths = [
            os.path.join("simulation_outputs","paper1","stairs","RANDOM_FB","no_learn_a","32.0","subject_expects_feedback_std_0.01_noClamp"),
            os.path.join("simulation_outputs","paper1","stairs","RANDOM_FB","learn_a_10","32.0","subject_expects_feedback_std_0.01_noClamp"),
            os.path.join("simulation_outputs","paper1","stairs","RANDOM_FB","learn_a_5.0","32.0","subject_expects_feedback_std_0.01_noClamp"),
            os.path.join("simulation_outputs","paper1","stairs","RANDOM_FB","learn_a_1.0","32.0","subject_expects_feedback_std_0.01_noClamp")
        ]
        namelist = [
            'k1_a = +oo',
            'k1_a = 10',
            'k1_a = 5',
            'k1_a = 1'
        ]
        exploit_saved_data(savepaths,namelist)
        exploit_saved_data_old(savepaths,namelist)

    def plot_abs_confidence():
        savepaths = [
            os.path.join("simulation_outputs","paper1","stairs","RANDOM_FB","no_learn_a","32.0","subject_expects_feedback_std_0.01_noClamp")
        ]
        namelist = [
            'k1_a = +oo'
        ]
        exploit_saved_data(savepaths,namelist)
        # exploit_saved_data_old(savepaths,namelist)

    plot_abs_confidence()
    input()