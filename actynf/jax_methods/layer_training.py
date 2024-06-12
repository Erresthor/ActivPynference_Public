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
    
from .jax_toolbox import _normalize,convert_to_one_hot_list,_swapaxes
from .layer_pick_action import sample_action_pyro

from .layer_options import DEFAULT_PLANNING_OPTIONS,DEFAULT_LEARNING_OPTIONS

from .layer_trial import synthetic_trial
from .layer_trial import empirical as empirical_trial

from .layer_learn import learn_after_trial

from actynf.jax_methods.shape_tools import to_log_space,vectorize_weights,get_vectorized_novelty


def training_step(trial_rng_key,T,
            pa,pb,pd,c,e,
            A_vec,B_vec,D_vec,U,
            selection_method="stochastic",alpha = 16, 
            planning_options = DEFAULT_PLANNING_OPTIONS,
            learning_options = DEFAULT_LEARNING_OPTIONS):
    """_summary_

    Args:
        trial_rng_key (_type_): _description_
        T (_type_): _description_
        Th (_type_): _description_
        pa (_type_): _description_
        pb (_type_): _description_
        pd (_type_): _description_
        c (_type_): _description_
        e (_type_): _description_
        A_vec (_type_): Vectorized process emission rule
        B_vec (_type_): Vectorized process transition rule
        D_vec (_type_): Vectorized process initial state rule
        selection_method (str, optional): _description_. Defaults to "stochastic".
        alpha (int, optional): _description_. Defaults to 16.
        planning_options (_type_, optional): _description_. Defaults to DEFAULT_PLANNING_OPTIONS.
        learn_dictionnary (_type_, optional): _description_. Defaults to DEFAULT_LEARNING_OPTIONS.

    Returns:
        _type_: _description_
    """
    
    # Vectorize the model weights : 
    trial_a,trial_b,trial_d = vectorize_weights(pa,pb,pd,U)
    trial_c,trial_e = to_log_space(c,e)
    trial_a_nov,trial_b_nov = get_vectorized_novelty(pa,pb,U,compute_a_novelty=True,compute_b_novelty=True)
    
    
    # T timesteps happen below : 
    [obs_darr,obs_arr,obs_vect_arr,
        true_s_darr,true_s_arr,true_s_vect_arr,
        u_d_arr,u_arr,u_vect_arr,
        qs_arr,qpi_arr,efes] = synthetic_trial(trial_rng_key,T,
                    A_vec,B_vec,D_vec,
                    trial_a,trial_b,trial_c,trial_d,trial_e,
                    trial_a_nov,trial_b_nov,
                    alpha = alpha,selection_method=selection_method,
                    planning_options=planning_options)
    
    # Then, we update the parameters of our HMM model at this level
    # We use the raw weights here !
    a_post,b_post,c_post,d_post,e_post,qs_post = learn_after_trial(obs_vect_arr,qs_arr,u_vect_arr,
                                            pa,pb,c,pd,e,U,
                                            learn_what=learning_options["bool"],
                                            learn_rates=learning_options["rates"],
                                            post_trial_smooth=learning_options["smooth_states"])
    
    return_tuple = ( obs_darr,obs_arr,obs_vect_arr,
                    true_s_darr,true_s_arr,true_s_vect_arr,
                    u_d_arr,u_arr,u_vect_arr,
                    qs_arr,qs_post,qpi_arr,efes,
                    a_post,b_post,c_post,d_post,e_post)
    
    return return_tuple

# Very fast methods
def synthetic_training(rngkey,
            Ntrials,T,
            A,B,D,U,
            a0,b0,c,d0,e,
            selection_method = "stochastic",alpha = 16, 
            planning_options = DEFAULT_PLANNING_OPTIONS,
            learning_options = DEFAULT_LEARNING_OPTIONS):
    normA,normB,normD = vectorize_weights(A,B,D,U)
        # These weights are the same across the whole training
    
    def _scan_training(carry,key):
        key,trial_key = jr.split(key)
        
        pa,pb,pd = carry
        
        # T timesteps happen below : 
        ( obs_darr,obs_arr,obs_vect_arr,
        true_s_darr,true_s_arr,true_s_vect_arr,
        u_d_arr,u_arr,u_vect_arr,
        qs_arr,qs_post,qpi_arr,efes,
        a_post,b_post,c_post,d_post,e_post) = training_step(trial_key,T,
            pa,pb,pd,c,e,
            normA,normB,normD,U,
            selection_method=selection_method,alpha = alpha, 
            planning_options = planning_options,
            learning_options = learning_options)
        
        # a_post,b_post,d_post = pa,pb,pd
        return (a_post,b_post,d_post),(
                    obs_darr,obs_arr,obs_vect_arr,
                    true_s_darr,true_s_arr,true_s_vect_arr,
                    u_d_arr,u_arr,u_vect_arr,
                    qs_arr,qs_post,qpi_arr,efes,
                    a_post,b_post,c_post,d_post,e_post)
        
    
    next_keys = jr.split(rngkey, Ntrials)
    (final_a,final_b,final_d), (
        all_obs_darr,all_obs_arr,all_obs_vect_arr,
        all_true_s_darr,all_true_s_arr,all_true_s_vect_arr,
        all_u_d_arr,all_u_arr,all_u_vect_arr,
        all_qs_arr,all_qs_post,all_qpi_arr,efes_arr,
        a_hist,b_hist,c_hist,d_hist,e_hist) = jax.lax.scan(_scan_training, (a0,b0,d0),next_keys)
    
    return [all_obs_arr,all_true_s_arr,all_u_arr,all_qs_arr,all_qs_post,all_qpi_arr,efes_arr,a_hist,b_hist,d_hist]

def synthetic_training_multi_subj(rngkeys_for_all_subjects,
            Ntrials,T,
            a0,b0,c,d0,e,
            A,B,D,U,
            selection_method="stochastic",alpha = 16, 
            planning_options = DEFAULT_PLANNING_OPTIONS,
            learning_options = DEFAULT_LEARNING_OPTIONS):

    map_this_function = partial(synthetic_training,
            Ntrials=Ntrials,T=T,
            a0=a0,b0=b0,c=c,d0=d0,e=e,
            A=A,B=B,D=D,U=U,
            selection_method = selection_method,alpha = alpha,
            planning_options = planning_options, 
            learning_options = learning_options)
    mapped_over_subjects = vmap(map_this_function)(rngkeys_for_all_subjects)
    return mapped_over_subjects


# Models used for fitting !
def empirical(obs_vect,act_vect,
        pa0,pb0,c,pd0,e,U,
        planning_options = DEFAULT_PLANNING_OPTIONS,
        learning_options = DEFAULT_LEARNING_OPTIONS):
    """,
    This method uses the compute_trial_posteriors_empirical function from the ai_jax_loop .py file
    It provides active inference agents with observation and returns their action posterior depending on their internal parameters.
    To allow for better convergence, the empirical actions at time t are observed at time t+1 instead of relying
    on computed action posteriors.
    
    Inputs : 
    - obs_vect : one_hot-encoded observations along a list of observation modalities. Each tensor in the list is of size Ntrials x Ntimesteps x Nobservations
    - act_vect : one_hot-encoded *observed* actions. This tensor is of size Ntrials x Ntimesteps x Nactions
    This method accounts for training wide effects by performing learning updates at the end of each trial.
    """
    
    def _scan_training(carry,data_trial):
        
        (pre_a,pre_b,pre_d) = carry
        
        (obs_trial,act_trial) = data_trial
        
        
        trial_a,trial_b,trial_d = vectorize_weights(pre_a,pre_b,pre_d,U)
        trial_c,trial_e = to_log_space(c,e)
        trial_a_nov,trial_b_nov = get_vectorized_novelty(pre_a,pre_b,U,compute_a_novelty=True,compute_b_novelty=True)
        
        # Empirical based state + action posterior for the whole trial
        qs_arr,qpi_arr = empirical_trial(obs_trial,act_trial,
                                trial_a,trial_b,trial_c,trial_d,trial_e,
                                trial_a_nov,trial_b_nov,
                                include_last_observation=True,
                                planning_options=planning_options)

        # NO ACTION SELECTION HERE ! We're using empirical observations 
        # to compute the parameters evolution instead
        
        # # Then, we update the parameters of our HMM model at this level
        a_post,b_post,_,d_post,_,qs_post = learn_after_trial(obs_trial,qs_arr,act_trial,
                                                 pre_a,pre_b,c,pre_d,e,U,
                                                 learn_what=learning_options["bool"],
                                                 learn_rates=learning_options["rates"],
                                                 post_trial_smooth=learning_options["smooth_states"])
        return (a_post,b_post,d_post),(qs_arr,qs_post,qpi_arr,a_post,b_post,d_post)
    
    final_matrices,(training_qs_arr,training_qs_post,training_qpi_arr,training_a_post,training_b_post,training_d_post) = jax.lax.scan(_scan_training,(pa0,pb0,pd0),(obs_vect,act_vect))
    
    return [training_qs_arr,training_qs_post,training_qpi_arr,training_a_post,training_b_post,training_d_post]


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