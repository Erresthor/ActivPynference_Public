import os
import random as ra
import numpy as np
import time
import copy
import matplotlib.pyplot as plt

from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp

from jax.tree_util import tree_map
from jax import lax,vmap,jit

import arviz as az
import corner 

from numpyro import plate,sample,deterministic
import numpyro.distributions as distr
import numpyro 
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive    
    
from fastprogress.fastprogress import progress_bar

from jax_toolbox import _normalize,convert_to_one_hot_list,_swapaxes
from actynf.jax_methods.layer_plan import sample_action_pyro

from actynf.jax_methods.layer_trial import synthetic_trial,compute_trial_posteriors_empirical

from actynf.jax_methods.layer_learn import learn_after_trial


def synthetic_training(rngkey,Ns,Nos,Np,Ntrials,T,
            a0,b0,c,d0,e,
            A,B,D,
            Th =3,
            selection_method="stochastic",alpha = 16,gamma = None, 
            learn_dictionnary = {"bool":{"a":True,"b":True,"d":True},"rates":{"a":1.0,"b":1.0,"d":1.0}},smooth_state_estimates=False):
    A = _normalize(A,tree=True)
    B,_ = _normalize(B)
    D,_ = _normalize(D)
    
    def _scan_training(carry,key):
        key,trial_key = jr.split(key)
        
        pa,pb,pd = carry
        
        # T timesteps happen below : 
        [obs_darr,obs_arr,obs_vect_arr,
         true_s_darr,true_s_arr,true_s_vect_arr,
         u_d_arr,u_arr,u_vect_arr,
         qs_arr,qpi_arr,efes] = synthetic_trial(trial_key,
                        Ns,Nos,Np,
                        pa,pb,c,pd,e,
                        A,B,D,
                        T= T,Th = Th,
                        alpha = alpha,gamma = gamma, 
                        selection_method=selection_method)
        
        # # Then, we update the parameters of our HMM model at this level
        # The learn function wants the T dimension to be the last one
        o_hist_learn = _swapaxes(obs_vect_arr,tree=True)
        s_hist_learn = _swapaxes(qs_arr)
        u_hist_learn = _swapaxes(u_vect_arr)
        a_post,b_post,d_post = learn_after_trial(o_hist_learn,s_hist_learn,u_hist_learn,
                                                 pa,pb,pd,
                                                 learn_what=learn_dictionnary["bool"],
                                                 learn_rates=learn_dictionnary["rates"],
                                                 post_trial_smooth=smooth_state_estimates)
        # a_post,b_post,d_post = pa,pb,pd
        return (a_post,b_post,d_post),(
                    obs_darr,obs_arr,obs_vect_arr,
                    true_s_darr,true_s_arr,true_s_vect_arr,
                    u_d_arr,u_arr,u_vect_arr,
                    qs_arr,qpi_arr,efes,
                    a_post,b_post,d_post)
        
    
    next_keys = jr.split(rngkey, Ntrials)
    (_,_,_), (
        all_obs_darr,all_obs_arr,all_obs_vect_arr,
        all_true_s_darr,all_true_s_arr,all_true_s_vect_arr,
        all_u_d_arr,all_u_arr,all_u_vect_arr,
        all_qs_arr,all_qpi_arr,efes_arr,
        a_hist,b_hist,d_hist) = jax.lax.scan(_scan_training, (a0,b0,d0),next_keys)
    
    return [all_obs_arr,all_true_s_arr,all_u_arr,all_qs_arr,all_qpi_arr,efes_arr,a_hist,b_hist,d_hist]

def synthetic_training_multi_subj(rngkeys_for_all_subjects,Ns,Nos,Np,Ntrials,T,
            a0,b0,c,d0,e,
            A,B,D,
            Th =3,
            selection_method="stochastic",alpha = 16,gamma = None, 
            learn_dictionnary = {"bool":{"a":True,"b":True,"d":True},"rates":{"a":1.0,"b":1.0,"d":1.0}},smooth_state_estimates=False):

    map_this_function = partial(synthetic_training,Ns=Ns,Nos=Nos,Np=Np,Ntrials=Ntrials,T=T,
            a0=a0,b0=b0,c=c,d0=d0,e=e,
            A=A,B=B,D=D,
            Th = Th,
            selection_method=selection_method,alpha = alpha,gamma = gamma, 
            learn_dictionnary = learn_dictionnary,smooth_state_estimates=smooth_state_estimates)
    
    mapped_over_subjects = vmap(map_this_function)(rngkeys_for_all_subjects)
    return mapped_over_subjects

def compute_training_posteriors_empirical(obs_vect,act_vect,
        pa0,pb0,c,pd0,e,
        Np,Th =3,gamma=None,
        learn_dictionnary = {"bool":{"a":True,"b":True,"d":True},"rates":{"a":1.0,"b":1.0,"d":1.0}},smooth_state_estimates=False):
    """,
    This method uses the compute_trial_posteriors_empirical function from the ai_jax_loop .py file
    It provides active inference agents with observation and returns their action posterior depending on their internal parameters.
    To allow for better convergence, the actions chosen by these agents
    Inputs : 
    - obs_vect : one_hot-encoded observations along a list of observation modalities. Each tensor in the list is of size Ntrials x Ntimesteps x Nobservations
    - act_vect : one_hot-encoded *observed* actions. This tensor is of size Ntrials x Ntimesteps x Nactions
    This method accounts for training wide effects by performing learning updates at the end of each trial.
    """
    
    def _scan_training(carry,data_trial):
        
        (pre_a,pre_b,pre_d) = carry
        
        (obs_trial,act_trial) = data_trial
        
        
        # Empirical based state + action posterior for the whole trial
        qs_arr,qpi_arr = compute_trial_posteriors_empirical(obs_trial,act_trial,
                                Np,
                                pre_a,pre_b,c,pre_d,e,
                                Th =Th,gamma = gamma,
                                include_last_observation=True)

        # NO ACTION SELECTION HERE ! We're using empirical observations instead
        
        # # Then, we update the parameters of our HMM model at the training level
        # *Warning !* The learn function wants the T dimension to be the last one, so let's swap the axes -1 and -2 for the recorded history
        o_hist_learn = _swapaxes(obs_trial,tree=True)
        s_hist_learn = _swapaxes(qs_arr)
        u_hist_learn = _swapaxes(act_trial)
        a_post,b_post,d_post = learn_after_trial(o_hist_learn,s_hist_learn,u_hist_learn,
                                                 pre_a,pre_b,pre_d,
                                                 learn_what=learn_dictionnary["bool"],
                                                 learn_rates=learn_dictionnary["rates"],
                                                 post_trial_smooth=smooth_state_estimates)
        
        return (a_post,b_post,d_post),(qs_arr,qpi_arr,a_post,b_post,d_post)
    
    final_matrices,(training_qs_arr,training_qpi_arr,training_a_post,training_b_post,training_d_post) = jax.lax.scan(_scan_training,(pa0,pb0,pd0),(obs_vect,act_vect))
    
    return [training_qs_arr,training_qpi_arr,training_a_post,training_b_post,training_d_post]

def _test_simulate_trial():
    T = 10
    Ntrials = 100
     
    Nos = [5,5]
    Ns = 5
    Np = 5

    
    # PROCESS CONSTANTS : 
    trueA = [_normalize(jnp.eye(Ns))[0], _normalize(jnp.eye(Ns))[0]]
    
    plow = 0.3
    pup = 1.0
    trueB = np.zeros((Ns,Ns,Np))
    for u in range(Np):
        for s in range(Ns):
            if s >0:
                trueB[s-1,s,u] = plow
                trueB[s,s,u] = 1-plow
            else : 
                trueB[s,s,u] = 1.0
        try :
            trueB[u+1,u,u] = pup
            trueB[u,u,u] = 1.0 - pup
            trueB[u-1,u,u] = 0.0
        except:
            lol = "lol"
    
    
    trueB,_ = _normalize(jnp.asarray(trueB))
    
    fig,axes = plt.subplots(1,Np)
    for k,ax in enumerate(axes):
        ax.imshow(trueB[:,:,k],vmin = 0,vmax = 1)
    fig.show()
    # input()
    
    trueD = jax.nn.one_hot(0,Ns) 
    

    # MODEL TRUE PARAMETERS : 
    key = jr.PRNGKey(np.random.randint(0,900))
    
    true_ka1 = 0.1
    true_ka2 = 0.2
    true_kb = 0.1
    
    init_a_conf = 1.0
    init_b_conf = 0.01
    
    
    par1 = [true_ka1,true_ka2]
    par2 = true_kb

    val_a = [init_a_conf*_normalize(parameter_m*A_m + (1-parameter_m)*jnp.ones(A_m.shape))[0] for (parameter_m,A_m) in zip(par1,trueA)]
    val_b = init_b_conf*_normalize(par2*trueB + (1-par2)*jnp.ones(trueB.shape))[0]
    val_d = copy.deepcopy(trueD)
    val_c = [jnp.zeros((a_mod.shape[0],)) for a_mod in val_a]
    val_c[0] = jnp.linspace(0,val_c[0].shape[0],val_c[0].shape[0])
    val_e = jnp.ones((Np,))
    
    Th = 3
    learn_what = {
        "bool":{"a":True,"b":True,"d":False},
        "rates":{"a":1.0,"b":1.0,"d":1.0}
    }
    alpha = 16
    gamma = None
    selection_method="stochastic"
    
    rngkey = key
    res = synthetic_training(rngkey,Ns,Nos,Np,Ntrials,T,
            val_a,val_b,val_c,val_d,val_e,
            trueA,trueB,trueD,
            Th =Th,
            selection_method=selection_method, alpha = alpha,gamma = gamma,
            learn_dictionnary = learn_what,smooth_state_estimates=False)
    
    [all_obs_arr,all_true_s_arr,all_u_arr,all_qs_arr,all_qpi_arr,efes_arr,a_hist,b_hist,d_hist] = res
    
    # print(all_qpi_arr)
    
    # print(all_u_arr)
    # print(np.round(np.array(efes_arr[0,...])))
    # print(np.round(np.array(efes_arr[1,...])))
    # print(np.round(jax.nn.softmax(efes_arr[0,...]),2))
    
    
    # for trial in range(Ntrials):
    #     for action in range(Np):
    #         print(np.round(np.array(b_hist[trial][...,action]),2))
    for trial in range(Ntrials):
        print(np.round(np.array(a_hist[0][trial]),2))
    # exit()
    
    # Plotting : 
    plot_every = 5
    Nplots=  Ntrials//plot_every
    fig1,axes = plt.subplots(Nplots,2)
    fig2,axes2 = plt.subplots(Nplots,2+Np)
    # print(Nplots)
    # print([trial for trial in range(Ntrials) if (trial%plot_every==0)])
    for kplot,plotted_trial in enumerate([trial for trial in range(Ntrials) if (trial%plot_every==0)]) :
        print(plotted_trial)
        axes[kplot,0].imshow(np.array(_swapaxes(all_qs_arr[plotted_trial,...])),vmin=0,vmax=1)
        axes[kplot,1].imshow(np.array(_swapaxes(all_qpi_arr[plotted_trial,...])),vmin=0,vmax=1)
        
        if plotted_trial>0:
            axes2[kplot,0].imshow(np.array(_normalize(a_hist[0][plotted_trial-1,...])[0]),vmin=0,vmax=1)
            axes2[kplot,1].imshow(np.array(_normalize(a_hist[1][plotted_trial-1,...])[0]),vmin=0,vmax=1)
        else : 
            axes2[kplot,0].imshow(np.array(_normalize(val_a[0])[0]),vmin=0,vmax=1)
            axes2[kplot,1].imshow(np.array(_normalize(val_a[1])[0]),vmin=0,vmax=1)
        
        for action in range(Np):
            ax = axes2[kplot,action+2]
            if plotted_trial>0:
                ax.imshow(np.array(_normalize(b_hist[plotted_trial-1,...])[0])[:,:,action],vmin=0,vmax=1)
            else :
                ax.imshow(np.array(_normalize(val_b)[0])[:,:,action],vmin=0,vmax=1)
    fig1.show()
    fig2.show()
    input()



if __name__ == "__main__":
    T = 10
    Ntrials = 20
     
    Nos = [5,5]
    Ns = 5
    Np = 5

    
    # PROCESS CONSTANTS : 
    trueA = [_normalize(jnp.eye(Ns))[0], _normalize(jnp.eye(Ns))[0]]
    
    plow = 0.3
    pup = 1.0
    trueB = np.zeros((Ns,Ns,Np))
    for u in range(Np):
        for s in range(Ns):
            if s >0:
                trueB[s-1,s,u] = plow
                trueB[s,s,u] = 1-plow
            else : 
                trueB[s,s,u] = 1.0
        try :
            trueB[u+1,u,u] = pup
            trueB[u,u,u] = 1.0 - pup
            trueB[u-1,u,u] = 0.0
        except:
            lol = "lol"
    
    
    trueB,_ = _normalize(jnp.asarray(trueB))
    
    # fig,axes = plt.subplots(1,Np)
    # for k,ax in enumerate(axes):
    #     ax.imshow(trueB[:,:,k],vmin = 0,vmax = 1)
    # fig.show()
    # input()
    
    trueD = jax.nn.one_hot(0,Ns) 
    

    # MODEL TRUE PARAMETERS : 
    key = jr.PRNGKey(np.random.randint(0,900))
    
    true_ka1 = 0.5
    true_ka2 = 0.4
    true_kb = 0.2
    
    init_a_conf = 10.0
    init_b_conf = 1.0
    
    
    par1 = [true_ka1,true_ka2]
    par2 = true_kb

    val_a = [init_a_conf*_normalize(parameter_m*A_m + (1-parameter_m)*jnp.ones(A_m.shape))[0] for (parameter_m,A_m) in zip(par1,trueA)]
    val_b = init_b_conf*_normalize(par2*trueB + (1-par2)*jnp.ones(trueB.shape))[0]
    val_d = copy.deepcopy(trueD)
    val_c = [jnp.zeros((a_mod.shape[0],)) for a_mod in val_a]
    val_c[0] = jnp.linspace(0,val_c[0].shape[0],val_c[0].shape[0])
    val_e = jnp.ones((Np,))
    
    Th = 2
    learn_what = {
        "bool":{"a":True,"b":True,"d":False},
        "rates":{"a":1.0,"b":1.0,"d":1.0}
    }
    alpha = 16
    gamma = None
    selection_method="stochastic"
    
    rngkey = key
    res = synthetic_training(rngkey,Ns,Nos,Np,Ntrials,T,
            val_a,val_b,val_c,val_d,val_e,
            trueA,trueB,trueD,
            Th =Th,
            selection_method=selection_method, alpha = alpha,gamma = gamma,
            learn_dictionnary = learn_what,smooth_state_estimates=False)
    
    
    # Simulated trial 
    [all_obs_arr,all_true_s_arr,all_u_arr,all_qs_arr,all_qpi_arr,efes_arr,a_hist,b_hist,d_hist] = res
    
    data = {
        "observations":all_obs_arr,
        "actions":all_u_arr
    }

    
    
    
    
    
    
    
    
    
    # exit()
    
    # INFERENCE -------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    
    def get_params(Z):
        # Ensure that the parameters are positive and smoothed towards the lower values
        pre_a_mod1 = jax.nn.sigmoid(Z[0]) # jax.nn.softplus(x1[0]) #par1[0] # 
        pre_a_mod2 = jax.nn.sigmoid(Z[1]) # par1[1] # jax.nn.softplus(x1[1])
        pre_b = jax.nn.sigmoid(Z[2]) #
        
        pre_a_mod1 = deterministic("ka1",pre_a_mod1)
        pre_a_mod2 = deterministic("ka2",pre_a_mod2)
        pre_b= deterministic("kb",pre_b)
        
        model_initial_a_conf = 10.0
        # Initialize corresponding model parameters : 
        model_a0 = [model_initial_a_conf*_normalize(parameter_m*A_m + (1-parameter_m)*jnp.ones(A_m.shape))[0] for (parameter_m,A_m) in zip([pre_a_mod1,pre_a_mod2],trueA)]
        
        model_initial_b_conf = 1.0
        model_b0 = model_initial_b_conf*_normalize(pre_b*trueB + (1-pre_b)*jnp.ones(trueB.shape))[0]
        model_d0 = jax.lax.stop_gradient(copy.deepcopy(val_d))
        model_c = jax.lax.stop_gradient(copy.deepcopy(val_c))
        model_e = jax.lax.stop_gradient(copy.deepcopy(val_e))
        
        Nos = tree_map(lambda mat : mat.shape[0],model_a0)
        Np = model_b0.shape[-1]
        return model_a0,model_b0,model_c,model_d0,model_e,Np,Nos
    
    
    def model(_data):
        # Parameter priors : 
        Z = sample("Theta",distr.Uniform(-3,3).expand([3]).to_event(1))
        
        _a0,_b0,_c,_d0,_e,_Np,_Nos = get_params(Z)
        
        _gamma = None
        _alpha = 16
        _Th = 2
        
        # emissions = deterministic("observation_t",_data["observations"])
        
        emissions = _data["observations"]
        
        observation_vector = convert_to_one_hot_list(emissions,_Nos)
        observed_actions = _data["actions"]
        print(observed_actions.shape)
        action_vector = jax.nn.one_hot(observed_actions,_Np)
        
        assumed_learn_dictionnary = {"bool":{"a":True,"b":True,"d":False},"rates":{"a":1.0,"b":1.0,"d":1.0}}

        [
            training_qs_arr,training_qpi_arr,
            training_a_post,training_b_post,training_d_post
        ] = compute_training_posteriors_empirical(observation_vector,action_vector,
                _a0,_b0,_c,_d0,_e,
                _Np,Th =_Th,gamma=_gamma,
                learn_dictionnary=assumed_learn_dictionnary,smooth_state_estimates=False)
        
        sample_action_pyro(training_qpi_arr,_Np,_alpha,selection_method = "stochastic_alpha",observed_action=observed_actions)
    
    # exit()
    # empirical posterior based on chosen simulation parameters : 
    
    # infer_ka1 = 0.1
    # infer_ka2 = 0.2
    # infer_kb = 0.1
    
    # infer_par1 = [true_ka1,true_ka2]
    # infer_par2 = true_kb
    
    # infer_a0 = [init_a_conf*_normalize(parameter_m*A_m + (1-parameter_m)*jnp.ones(A_m.shape))[0] for (parameter_m,A_m) in zip(infer_par1,trueA)]
    # infer_b0 = init_b_conf*_normalize(infer_par2*trueB + (1-infer_par2)*jnp.ones(trueB.shape))[0]
    # infer_d0 = copy.deepcopy(trueD)
    
    # vect_observation_full_training = convert_to_one_hot_list(all_obs_arr,Nos)
    # vect_actions_full_training = jax.nn.one_hot(all_u_arr,Np)
    
    # [emp_training_qs_arr,emp_training_qpi_arr,emp_training_a_post,emp_training_b_post,emp_training_d_post] = compute_training_posteriors_empirical(vect_observation_full_training,vect_actions_full_training,
    #     Np,
    #     infer_a0,infer_b0,val_c,infer_d0,val_e,
    #     Th =3,
    #     learn_dictionnary = {"bool":{"a":True,"b":True,"d":True},"rates":{"a":1.0,"b":1.0,"d":1.0}},smooth_state_estimates=False)
    
    os.environ["PATH"] += os.pathsep + 'C:\\Users\\annic\\OneDrive\\Bureau\\MainPhD\\code\\venvs\\Graphviz\\bin'
    graph = numpyro.render_model(model, model_args=(data,), filename="model.pdf",render_distributions=True,render_params=True)    
    
    with handlers.seed(rng_seed=123):
        model(data)
    
    tstart = time.time()
    pred_samples = Predictive(model,num_samples=15)(key,data)
    tend = time.time()
    print(pred_samples)
    print("Elapsed : "+str(tend-tstart))
    
    
    num_warmup, num_samples = 1000, 1000
    mcmc = MCMC(NUTS(model=model), num_warmup=num_warmup, num_samples=num_samples,num_chains=1)
    mcmc.run(jr.PRNGKey(2), data) 
    
    posterior_samples = mcmc.get_samples()
    print(posterior_samples)
    mcmc.print_summary()

    data_mcmc = az.from_numpyro(posterior = mcmc)
    az.plot_trace(data_mcmc, compact=True, figsize=(15, 25));
    plt.show()
    
    inf_data = az.from_numpyro(mcmc)
    az.summary(inf_data)
    
    corner.corner(inf_data, var_names=["ka1", "ka2","kb"], truths=[true_ka1,true_ka2,true_kb])
    plt.show()