import sys,os
import time
import numpy as np
import pickle 
import matplotlib.pyplot as plt

import actynf
from utils.basic_task import build_training_process,build_subject_model

import jax
from actynf.jax_methods.layer_training import synthetic_training_multi_subj

def save_output(stm_subjs,weight_subjs,savepath,overwrite=False):
    exists = os.path.isfile(savepath)
    if (not(exists)) or (overwrite):
        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
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



# Plotting helpers
def clever_running_mean(arr, N):
    """ 
    For regularly spaced points only
    """    
    xarr = np.array(arr)
    xpost = np.zeros(xarr.shape)
    # raw_conv = np.convolve(x, np.ones(N)/N, mode='same')
    for k in range(xarr.shape[0]):
        localmean = 0.0
        cnt = 0.0
        for i in range(k-N,k+N+1,1):
            if ((i>=0) and (i<xarr.shape[0])):
                localmean += xarr[i]
                cnt += 1
        xpost[k] = localmean/(cnt+1e-18)
    return xpost

def plot_trials(state_arr,feedback_arr,intero_arr,Ns,window = 1):
    # Mean of all subjects
    fig1,axes1 = plt.subplots(2,1,dpi=180)
    colorlist = ["red","blue"]
    labelist = ["External","Internal"]

    # State 1 : 
    f = 0
    axes_states = axes1[0]
    color = 'green'
    states = state_arr[:,:,:]
    flat_state_f = states.reshape(states.shape[0],-1)
    m_state = clever_running_mean(np.mean(flat_state_f,axis=0),window)
    v_state = clever_running_mean(np.std(flat_state_f,axis=0),window)
    xs = np.linspace(0,m_state.shape[0],m_state.shape[0])
    axes_states.fill_between(xs,m_state-v_state,m_state+v_state,color=color,alpha=0.2)
    p1, = axes_states.plot(xs,m_state,color=color,label = 'mental state')
    axes_states.set_ylim([-0.1,Ns[0]-1+0.1])
    axes_states.set_ylabel("Mental state")
    axes_states.set_xlabel("Timesteps")

    
    # Observations 1 & 2 : feedback and interoceptive  : 
    f = 0
    axes_fb1 = axes1[1]
    color = colorlist[f]
    feedback_arr = states.reshape(feedback_arr.shape[0],-1)
    m_state = clever_running_mean(np.mean(feedback_arr,axis=0),window)
    v_state = clever_running_mean(np.std(feedback_arr,axis=0),window)
    xs = np.linspace(0,m_state.shape[0],m_state.shape[0])
    axes_fb1.fill_between(xs,m_state-v_state,m_state+v_state,color=color,alpha=0.2)
    p1, = axes_fb1.plot(xs,m_state,color=color,label = labelist[f])
    axes_fb1.set_ylim([-0.1,Ns[0]-1+0.1])
    axes_fb1.set_ylabel("External feedback")
    axes_fb1.set_xlabel("Timesteps")
    
    axes_fb2 = axes_fb1.twinx()
    f = 1
    color = colorlist[f]
    intero_arr = states.reshape(intero_arr.shape[0],-1)
    m_state = clever_running_mean(np.mean(intero_arr,axis=0),window)
    v_state = clever_running_mean(np.std(intero_arr,axis=0),window)
    xs = np.linspace(0,m_state.shape[0],m_state.shape[0])
    axes_fb2.fill_between(xs,m_state-v_state,m_state+v_state,color=color,alpha=0.2)
    axes_fb2.axhline(y=0,color="black")
    p2, = axes_fb2.plot(xs,m_state,color=color,label = labelist[f])
    axes_fb2.set_ylim([-0.1,Ns[0]-1+0.1])
    axes_fb2.set_ylabel("Internal observations")
    axes_fb1.grid()
    
    axes_fb1.yaxis.label.set_color(p1.get_color())
    axes_fb2.yaxis.label.set_color(p2.get_color())
    axes_fb2.spines["right"].set_edgecolor(p2.get_color())
    axes_fb1.tick_params(axis='y', colors=p1.get_color())
    axes_fb2.tick_params(axis='y', colors=p2.get_color())
    

    fig1.tight_layout()
    # fig1.suptitle("Simulated Mental Imagery training with high prior mental imagery knowledge",y=1.0)
    fig1.show()









if __name__=='__main__':
    
    Nsubjects = 5
    Ntrials = 1
        
    # Environment variables
    Ns = 5
    
    
    
    
    
    # Subject variables
    _alpha = 1600
    rs = 10
    p_up = 1.0
    p_low = 0.0
    kas = [1.0,0.0]
    A,B,D,U = build_training_process(Ns,p_up,p_low,kas)
    # print(B)
    
    
    kas_subj =  [0.5,0.0]
    a_str = [10,100]
    kb_subj = 1.0
    b_str = 1.0
    kd = 0.2
    a,b,c,d,e,u = build_subject_model(Ns,
                        a_str,kas_subj,
                        B,kb_subj,b_str,
                        U,
                        kd,
                        rs)
    
    # -------------------------------------------------------------------------------------
    # Classical actynf network building :
    T = 10
    Th = 2
    process_layer = actynf.layer("process","process",
                 A,B,None,D,None,
                 U,T,Th)    
    
    model_layer = actynf.layer("model","model",
                 a,b,c,d,e,
                 U,T,Th)
    # model_layer.hyperparams.cap_state_explo = 3
    model_layer.hyperparams.b_novelty = True
    model_layer.learn_options.learn_a = True
    model_layer.learn_options.learn_b = True
    model_layer.learn_options.learn_c = False
    model_layer.learn_options.learn_d = True
    model_layer.learn_options.learn_e = False
    model_layer.hyperparams.alpha = _alpha
    
    process_layer.inputs.u = actynf.link(model_layer,lambda x : x.u)
    model_layer.inputs.o = actynf.link(process_layer, lambda x : x.o)
    net = actynf.layer_network([process_layer,model_layer],"training_model") 
    print(d)
    
    overwrite = True
    SAVING_FOLDER = os.path.join("simulation_outputs","simulations_jax")
    savepath = os.path.join(SAVING_FOLDER,"classic_simple.simu")
    exists = os.path.isfile(savepath)
    if (not(exists)) or (overwrite):
        t0 = time.time()
        result_stms = []
        results_weights = []
        for sub in range(Nsubjects):
            print("Subject " + str(sub) + " / " + str(Nsubjects))
            subj_net = net.copy_network(sub)
            STMs,weights = subj_net.run_N_trials(Ntrials)
            result_stms.append(STMs)
            results_weights.append(weights)
        t1 = time.time()
        print("Computations took "  + str(t1-t0) + " secs.")
        save_output(result_stms,results_weights,savepath,True)
    stms,weights,Nsubj,Ntrials = extract_training_data(savepath)
    
    
    print(a)
    # -------------------------------------------------------------------------------------
    # Jax equivalent : 
    
    random_keys = jax.random.PRNGKey(0)
    rng_all_subjects = jax.random.split(random_keys,Nsubjects)
    [all_obs_arr,all_true_s_arr,all_u_arr,all_qs_arr,all_qpi_arr,efes_arr,a_hist,b_hist,d_hist] =synthetic_training_multi_subj(
        rng_all_subjects,Ns,[Ns,Ns],u.shape[0],Ntrials,T,
        a,b[0],c,d[0],e,A,B[0],D[0],Th,alpha=_alpha
    )
    
    arr_qs_classic = np.swapaxes(np.array([[stms[subj][trial][1].x_d for trial in range(1,len(stms[subj]))] for subj in range(Nsubj)]),-1,-2)
    arr_u_classic = np.array([[stms[subj][trial][0].u for trial in range(1,len(stms[subj]))] for subj in range(Nsubj)])
    print(arr_qs_classic.shape)
    print(np.round(arr_qs_classic-all_qs_arr,3)[0])
    print(np.round(arr_qs_classic,3)[0])
    print(np.round(all_qs_arr ,3)[0])
    print(np.round(arr_u_classic ,3)[0])
    exit()
    jax_states = np.array(all_true_s_arr)
    jax_feedback = all_obs_arr[0]
    jax_intero = all_obs_arr[1]
    
    
    arr_feedback_levels = np.array([[stms[subj][trial][0].o[0,:] for trial in range(1,len(stms[subj]))] for subj in range(Nsubj)])
        # All the measured feedbacks ! (modality 0 : laterality, modality 1 : nf)
    arr_intero_levels = np.array([[stms[subj][trial][1].o[1,:] for trial in range(1,len(stms[subj]))] for subj in range(Nsubj)])
        # All the measured feedbacks ! (modality 0 : laterality, modality 1 : nf)
        
        
    arr_state_levels = np.array([[stms[subj][trial][0].x[0,:] for trial in range(1,len(stms[subj]))] for subj in range(Nsubj)])
            # All the true cognitive states ! (factor 0 : orientation of MI ERDs, factor 1 : intensity of MI ERDs)
    arr_state_belief_levels = np.array([[stms[subj][trial][1].x_d for trial in range(1,len(stms[subj]))] for subj in range(Nsubj)])
            # All the true cognitive states ! (factor 0 : orientation of MI ERDs, factor 1 : intensity of MI ERDs)

    # print(arr_state_levels.shape)
    print(arr_intero_levels.shape)
    plot_trials(arr_state_levels,arr_feedback_levels,arr_intero_levels,[Ns],1)
    plot_trials(jax_states,jax_feedback,jax_intero,[Ns],1)
    input()
    
    