import os
import random as ra
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
import arviz as az
import corner  

from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.tree_util import tree_map
from jax import lax, vmap, jit

from numpyro import plate,sample,deterministic
import numpyro.distributions as distr
import numpyro 
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive   
    
from .jax_toolbox import _normalize,_jaxlog,convert_to_one_hot_list
from .jax_toolbox import compute_novelty

from .layer_process import initial_state_and_obs,process_update
from .layer_infer_state import compute_state_posterior
from .layer_plan import policy_posterior,sample_action,sample_action_pyro

def compute_step_posteriors(t,prior,observation,
                            a,b,c,e,a_novel,b_novel,gamma,Np,Th):   
    # State inference
    qs = compute_state_posterior(prior,observation,a)
    # Policy planning
    efe,raw_qpi = policy_posterior(t,Th,qs,a,b,c,e,a_novel,b_novel,gamma,Np)
    return qs,raw_qpi,efe

def synthetic_trial(rngkey,
              Ns,Nos,Np,
              pa,pb,c,pd,e,
              A,B,D,
              T=10,Th =3,
              alpha = 16,gamma = None, selection_method="stochastic"):
    
    # Normalize the subject priors ( = get their expected values 
    # given the entertained dirichlet prior)
    a = _normalize(pa,tree=True)
    b,_ = _normalize(pb)
    d,_ = _normalize(pd)

    # Compute the prior novelty
    a_novel = compute_novelty(pa,True)
    b_novel = compute_novelty(pb)
    
    rngkey, init_key = jr.split(rngkey)
    
    # Initialize process
    [s_0_d,s_0_idx,s_0_vect],[o_0_d,o_0_idx,o_0_vect] = initial_state_and_obs(init_key,A,D,Ns,Nos)
    # Initialize subject model
    ps_0 = d

    def _scan(carry,xs):
        (key,t) = xs
        
        key,key_agent,key_process = jr.split(key,3)  # For random generations
        
        # Saved states from previous process tick and model update (t-1) --------
        true_s,observation,prior = carry
        
        # ---------------------------------------------------------------------------------
        # Model update (t) ----------------------------------------------------------------
        
        # State & policy inference
        qs,raw_qpi,efe = compute_step_posteriors(t,prior,observation,a,b,c,e,a_novel,b_novel,gamma,Np,Th)
        
        # Action sampling
        u_d,u_idx,u_vect = sample_action(raw_qpi,Np,alpha, selection_method=selection_method,rng_key=key_agent)
        
        # Prior for next timestep
        new_prior = jnp.einsum("iju,j,u->i",b,qs,u_vect)

        # ---------------------------------------------------------------------------------
        # Process update (t+1) -------------------------------------------------- 
        [s_d,s_idx,s_vect],[o_d,o_idx,o_vect] = process_update(key_process,true_s,A,B,u_vect,Ns,Nos)
        return (s_vect,o_vect,new_prior),(o_d,o_idx,o_vect,s_d,s_idx,s_vect,u_d,u_idx,u_vect,qs,raw_qpi,efe)
    
    timestamps = jnp.arange(T-1)
    next_keys = jr.split(rngkey, T - 1)
    (last_true_s,last_obs,last_prior), (obs_darr,obs_arr,obs_vect_arr,true_s_darr,true_s_arr,true_s_vect_arr,u_d_arr,u_arr,u_vect_arr,qs_arr,qpi_arr,efes) = jax.lax.scan(_scan, (s_0_vect,o_0_vect,ps_0),(next_keys,timestamps))
    
    # Compute the state posterior for the ultimate timestep
    last_qs = compute_state_posterior(last_prior,last_obs,a)

    # Don't forget the first elements !
    obs_darr = tree_map(lambda x,y :jnp.concatenate([x.reshape(1,-1),y],axis=0),o_0_d, obs_darr)
    obs_arr = tree_map(lambda x,y :jnp.concatenate([jnp.expand_dims(x,axis=0),y],axis=0),o_0_idx, obs_arr)
    obs_vect_arr = tree_map(lambda x,y :jnp.concatenate([x.reshape(1,-1),y],axis=0),o_0_vect, obs_vect_arr)
    
    true_s_darr = jnp.concatenate([s_0_d.reshape(1,-1),true_s_darr],axis=0)
    true_s_arr = jnp.concatenate([jnp.expand_dims(s_0_idx,axis=0),true_s_arr],axis=0)
    true_s_vect_arr = jnp.concatenate([s_0_vect.reshape(1,-1),true_s_vect_arr],axis=0)
    
    # And the last inference :
    qs_arr = jnp.concatenate([qs_arr,last_qs.reshape(1,-1)],axis=0)

    return [obs_darr,obs_arr,obs_vect_arr,true_s_darr,true_s_arr,true_s_vect_arr,u_d_arr,u_arr,u_vect_arr,qs_arr,qpi_arr,efes]


# STATE AND ACTION POSTERIOR IN RESPONSE TO EMPIRICAL OBSERVATION(S) + PREVIOUS ACTION(S)
def compute_trial_posteriors_empirical(obs_vect,act_vect,
        Np,
        pa,pb,c,pd,e,
        Th =3,gamma = None,include_last_observation=False):
    """ 
    This method compares observed actions and what the specified model would have done given a specific observation.
    
    Provide the model with a sequence of observations and ask for policy posteriors corresponding to the current situation.
    Note that the agent doesn't actually realize any actions despite its planning. 
    
    After inference, the true action realized is revealed to the agent. 
    """
    # Normalize the subject priors
    a = _normalize(pa,tree=True)
    b,_ = _normalize(pb)
    d,_ = _normalize(pd)

    # Compute the prior novelty
    a_novel = compute_novelty(pa,True)
    b_novel = compute_novelty(pb)
    
    def _scan(carry,data_t):
        emp_prior = carry
        (observation_t,observed_action_t_vect,t) = data_t
        
        qs,raw_qpi,efe = compute_step_posteriors(t,emp_prior,observation_t,a,b,c,e,a_novel,b_novel,gamma,Np,Th)
        
        # action_t_vect = action_t
        next_emp_prior = jnp.einsum("iju,j,u->i",b,qs,observed_action_t_vect)

        return next_emp_prior,(qs,raw_qpi)
    
    timestamps = jnp.arange(T-1)
    all_obs_but_last = tree_map(lambda x : x[:-1,...],obs_vect)
    last_prior,(qs_arr,qpi_arr) = jax.lax.scan(_scan,d,(all_obs_but_last,act_vect,timestamps))
    
    if include_last_observation : # Useful if we want to learn weights :)
        last_obs = tree_map(lambda x : x[-1,...],obs_vect)
        last_posterior = compute_state_posterior(last_prior,last_obs,a)
        qs_arr = jnp.concatenate([qs_arr,last_posterior.reshape(1,-1)],axis=0)
    
    return qs_arr,qpi_arr

if __name__=="__main__":
    T = 20
    
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
    fig,axes = plt.subplots(1,Np)
    for k,ax in enumerate(axes):
        ax.imshow(trueB[:,:,k],vmin = 0,vmax = 1)
    # fig.show()
    # input()
    
    trueB,_ = _normalize(jnp.asarray(trueB))
    
    trueD = jax.nn.one_hot(0,Ns) 
    

    # MODEL TRUE PARAMETERS : 
    key = jr.PRNGKey(np.random.randint(0,900))
    
    true_ka1 = 0.8
    true_ka2 = 0.1
    true_kb = 0.5
    
    
    par1 = [true_ka1,true_ka2]
    par2 = true_kb
    val_a = [1*_normalize(parameter_m*A_m + (1-parameter_m)*jnp.ones(A_m.shape))[0] for (parameter_m,A_m) in zip(par1,trueA)]
    val_b = 1*_normalize(par2*trueB + (1-par2)*jnp.ones(trueB.shape))[0]
    val_d = copy.deepcopy(trueD)
    val_c = [jnp.zeros((a_mod.shape[0],)) for a_mod in val_a]
    val_c[0] = jnp.linspace(0,val_c[0].shape[0],val_c[0].shape[0])
    val_e = jnp.ones((Np,))


    [obs_darr,obs_arr,obs_vect_arr,
    true_s_darr,true_s_arr,true_s_vect_arr,
    u_d_arr,u_arr,u_vect_arr,
    qs_arr,qpi_arr,efes] = synthetic_trial(key,
                    Ns,Nos,Np,
                    val_a,val_b,val_c,val_d,val_e,
                    trueA,trueB,trueD,
                    T,Th =3,alpha=16) 
    
    
    data = {
        "observations":obs_arr,
        "actions":u_arr
    }
    
    # emissions = deterministic("observation_t",data["observations"])
        
    # observation_vector = convert_to_one_hot_list(emissions,Nos)
    # observed_actions = data["actions"]
    # action_vector = jax.nn.one_hot(observed_actions,Np)
    # # print(qpi_arr)
    # print(qs_arr)
    # qs_arr,qpi_arr = compute_trial_posteriors_empirical(observation_vector,action_vector,
    #                     Np,
    #                     val_a,val_b,val_c,val_d,val_e,
    #                     Th = 3,gamma = None,
    #                     include_last_observation=True)
    
    # # print(qpi_arr)
    # print(qs_arr)
    # exit()

    def get_params(Z):
        # Ensure that the parameters are positive and smoothed towards the lower values
        pre_a_mod1 = jax.nn.sigmoid(Z[0]) # jax.nn.softplus(x1[0]) #par1[0] # 
        pre_a_mod2 = jax.nn.sigmoid(Z[1]) # par1[1] # jax.nn.softplus(x1[1])
        pre_b = jax.nn.sigmoid(Z[2]) #
        
        pre_a_mod1 = deterministic("ka1",pre_a_mod1)
        pre_a_mod2 = deterministic("ka2",pre_a_mod2)
        pre_b= deterministic("kb",pre_b)
        
        # Initialize corresponding model parameters : 
        model_a = [10*_normalize(parameter_m*A_m + (1-parameter_m)*jnp.ones(A_m.shape))[0] for (parameter_m,A_m) in zip([pre_a_mod1,pre_a_mod2],trueA)]
        model_b = 10*_normalize(pre_b*trueB + (1-pre_b)*jnp.ones(trueB.shape))[0]
        model_d = jax.lax.stop_gradient(copy.deepcopy(val_d))
        model_c = jax.lax.stop_gradient(copy.deepcopy(val_c))
        model_e = jax.lax.stop_gradient(copy.deepcopy(val_e))
        
        Nos = tree_map(lambda mat : mat.shape[0],model_a)
        Np = model_b.shape[-1]
        return model_a,model_b,model_c,model_d,model_e,Np,Nos
        
    
    def model(_data):
        # Parameter priors : 
        # Z = sample("Theta",distr.Normal(0.,1.0).expand([3]).to_event(1))
        Z = sample("Theta",distr.Uniform(-3,3).expand([3]).to_event(1))
        _a,_b,_c,_d,_e,_Np,_Nos = get_params(Z)
        
        _gamma = None
        _alpha = 16
        _Th = 3
        
        emissions = _data["observations"] # deterministic("observations_trial",_data["observations"])
        
        
        
        observation_vector = convert_to_one_hot_list(emissions,_Nos)
        
        observed_actions = _data["actions"]
        empirical_action_vector = jax.nn.one_hot(observed_actions,_Np)
        
        qs_arr,qpi_arr = compute_trial_posteriors_empirical(observation_vector,empirical_action_vector,
                        _Np,
                        _a,_b,_c,_d,_e,
                        Th = _Th,gamma = _gamma)
        
        sample_action_pyro(qpi_arr,_Np,_alpha,selection_method = "stochastic_alpha",observed_action=observed_actions)
    
    os.environ["PATH"] += os.pathsep + 'C:\\Users\\annic\\OneDrive\\Bureau\\MainPhD\\code\\venvs\\Graphviz\\bin'
    graph = numpyro.render_model(model, model_args=(data,), filename="model.pdf",render_distributions=True,render_params=True)    
    
    with handlers.seed(rng_seed=123):
        model(data)
    
    tstart = time.time()
    pred_samples = Predictive(model,num_samples=15)(key,data)
    tend = time.time()
    print(pred_samples)
    print("Elapsed : "+str(tend-tstart))
    
    
    # sample_vals = pred_samples(key,)
    
    
    # exit()
    # MCMC to estimate posterior parameter distribution
    num_warmup, num_samples = 200, 200
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