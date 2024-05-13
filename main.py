import numpy as np

import jax
import jax.numpy as jnp

import actynf
print("Actynf version : " + str(actynf.__version__))
from actynf.jax_methods.layer_training import synthetic_training_multi_subj
from actynf.jax_methods.planning_tools import autoexpand_preference_matrix
from demos.jax_demos.utils.basic_task import build_training_process,build_subject_model

if __name__ == '__main__': 
    # Environment variables
    Ns = 5
    Nsubjects = 1
    Ntrials = 200
    T = 10
    Th = 5
    
    # Subject variables
    _alpha = 4
    rs = 10
    p_up = 1.0
    p_low = 0.3
    kas = [0.999,0.0]
    A,B,D,U = build_training_process(Ns,p_up,p_low,kas)
    
    kas_subj =  [0.9,0.0]
    a_str = [30,100]
    kb_subj = 0.1
    b_str = 1.0
    kd = 0.2
    a,b,c,d,e,u = build_subject_model(Ns,
                        a_str,kas_subj,
                        B,kb_subj,b_str,
                        U,
                        kd,
                        rs)
    
    for k,cm in enumerate(c): 
        cm = np.array(jnp.repeat(jnp.expand_dims(cm,-1),T,-1))
        cm[:,:-1] = 0.0        
        c[k] = jnp.array(cm)
    c = autoexpand_preference_matrix(c,Th,"last")
    
    random_keys = jax.random.PRNGKey(2)
    rng_all_subjects = jax.random.split(random_keys,Nsubjects)
    [all_obs_arr,all_true_s_arr,all_u_arr,all_qs_arr,all_qpi_arr,efes_arr,a_hist,b_hist,d_hist] =synthetic_training_multi_subj(
        rng_all_subjects,Ns,[Ns,Ns],u.shape[0],Ntrials,T,
        a,b[0],c,d[0],e,A,B[0],D[0],Th,alpha=_alpha
    )
    
    print(all_obs_arr[0][0,100:,:])