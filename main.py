import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax,vmap, jit
from jax.tree_util import tree_map
from functools import partial,reduce


import actynf
print("Actynf version : " + str(actynf.__version__))
from actynf.jax_methods.layer_training import synthetic_training_multi_subj
from actynf.jax_methods.layer_options import get_planning_options
from actynf.jax_methods.planning_tools import autoexpand_preference_matrix

from actynf.jax_methods.jax_toolbox import _normalize

from actynf.jax_methods.planning_tools import compute_novelty

from demos.jax_demos.utils.basic_task import build_training_process,build_subject_model

from actynf.jax_methods.layer_trial import synthetic_trial,synthetic_trial_set_vals,synthetic_trial_direct_run,compute_step_posteriors,end_of_trial_filter

from actynf.jax_methods.layer_process import fetch_outcome,initial_state_and_obs

from actynf.demo_tools.tmaze.weights import get_T_maze_gen_process,get_T_maze_model
# def kronecker_prod_action(b_normed):
    
from actynf.jax_methods.layer_infer_state import compute_state_posterior
from actynf.jax_methods.layer_learn import learn_after_trial,backwards_pass
from actynf.jax_methods.jax_toolbox import _normalize,convert_to_one_hot_list,_swapaxes
from actynf.jax_methods.layer_options import DEFAULT_PLANNING_OPTIONS,DEFAULT_LEARNING_OPTIONS,get_learning_options


from actynf.jax_methods.shape_tools import to_log_space,vectorize_weights,get_vectorized_novelty


def example():
    A0 = jnp.array([
        [0.9,0.1],
        [0.1,0.9]
    ])
    A1 = jnp.array([
        [0.1,0.9],
        [0.9,0.1]
    ])
    A = [A0,A1]
    
    B0 = jnp.array([
        [0.1,0.9],
        [0.9,0.1]
    ])
    B1 = jnp.array([
        [0.9,0.1],
        [0.1,0.9]
    ])
    B = jnp.stack([B0,B1],-1)
    # print(B[...,0])
    # print(B[...,1])
    # exit()
    
    D = jnp.array([0.1,0.9])
    
    C = [jnp.array([0.0,0.0]),jnp.array([0.0,10.0])]
    # for mod in range(len(C)):
    #     C[mod] = jnp.expand_dims(C[mod],)
    
    
    s = jnp.array([0.8,0.2])
    
    key = jax.random.PRNGKey(6666)

    # res = fetch_outcome(key,2,[2,2],
    #         0,jnp.array([0.6,0.4]),jnp.array([1,0]),
    #         A,B,D,
    #         fixed_states_array=None,fixed_outcomes_tree=[None,None])

    # print(res)
    
    Np = B.shape[-1]
    
    
    
    static_states = jnp.array([0,-1,-1,-1,0,1,-1,-1,-1,1])
    
    qss,qpis,efes,states,obs = synthetic_trial_set_vals(key,
              2,[2,2],Np,
              [10*A0,10*A1],10*B,C,D,jnp.ones((Np,)),
              A,B,D,
              static_set_states = static_states,static_set_obs = None,
              T=10,Th =3,
              alpha = 16,gamma = None, selection_method="stochastic")
    
    print(qpis)
    print(qss)

def run_training_multi_factors(rngkey,Ns,Nos,Np,
            Ntrials,T,
            inA,inB,inD,inU,
            a0,b0,c,d0,e,
            Th =3,
            selection_method="stochastic",alpha = 16,gamma = None, 
            planning_options=DEFAULT_PLANNING_OPTIONS,
            learn_dictionnary = DEFAULT_LEARNING_OPTIONS):
    """
    Contrary to previous methods, this accepts multiple hidden factors ! :D 
    
    1. We use kronecker products to transform those multiple hidden dimensions into a single one (vec space)
    2. We make  computations with this single hidden dimension
    3. We flip back to the "source space" after the trial to update the B and D matrices !
    """    
    normA,normB,normD = vectorize_weights(inA,inB,inD,inU)
        # These weights are the same across the whole training
    
    def _scan_training(carry,key):
        key,trial_key = jr.split(key)
        
        pa,pb,pc,pd,pe = carry  # Dirichlet weights for this trial
        
        # From dirichlet weights to mdp parameters !
        trial_a,trial_b,trial_d = vectorize_weights(pa,pb,pd,inU)  # normalize & vectorize !
        trial_c,trial_e = to_log_space(pc,pe)
        
        trial_a_nov,trial_b_nov = get_vectorized_novelty(pa,pb,inU,compute_a_novelty=True,compute_b_novelty=True)
                    # get novelty scores, if needed
                    
        # T timesteps happen below : 
        [obs_darr,obs_arr,obs_vect_arr,
         true_s_darr,true_s_arr,true_s_vect_arr,
         u_d_arr,u_arr,u_vect_arr,
         qs_arr,qpi_arr,efes] = synthetic_trial_direct_run(trial_key,
                            Ns,Nos,Np,
                            trial_a,trial_a_nov,
                            trial_b,trial_b_nov,
                            trial_c,trial_d,trial_e,
                            normA,normB,normD,
                            T=T,Th =Th,
                            alpha = alpha, selection_method=selection_method,
                            planning_options=planning_options)
        
        a_post,b_post,c_post,d_post,e_post = learn_after_trial(obs_vect_arr,qs_arr,u_vect_arr,
                                                 pa,pb,pc,pd,pe,inU,
                                                 learn_what=learn_dictionnary["bool"],
                                                 learn_rates=learn_dictionnary["rates"],
                                                 post_trial_smooth=learn_dictionnary["smooth_states"])
        
        # a_post,b_post,d_post = pa,pb,pd
        return (a_post,b_post,c_post,d_post,e_post),(
                    obs_darr,obs_arr,obs_vect_arr,
                    true_s_darr,true_s_arr,true_s_vect_arr,
                    u_d_arr,u_arr,u_vect_arr,
                    qs_arr,qpi_arr,efes,
                    a_post,b_post,c_post,d_post,e_post)
        
    
    next_keys = jr.split(rngkey, Ntrials)
    (_,_,_,_,_), (
        all_obs_darr,all_obs_arr,all_obs_vect_arr,
        all_true_s_darr,all_true_s_arr,all_true_s_vect_arr,
        all_u_d_arr,all_u_arr,all_u_vect_arr,
        all_qs_arr,all_qpi_arr,efes_arr,
        a_hist,b_hist,c_hist,d_hist,e_hist) = jax.lax.scan(_scan_training, (a0,b0,c,d0,e),next_keys)
    
    return c_hist
    # return [all_obs_vect_arr,all_true_s_arr,all_u_vect_arr,all_qs_arr,all_qpi_arr,efes_arr,a_hist,b_hist,d_hist]
    return [all_obs_arr,all_true_s_arr,all_u_arr,all_qs_arr,all_qpi_arr,efes_arr,a_hist,b_hist,d_hist]

def learn_example():
    # Environment variables
    Nsubjects = 1
    Ntrials = 10
    T = 3
    Th = 3
    
    pHA = 0.8
    pWin = 0.99
    A,B,C,D,E,U = get_T_maze_gen_process(0.8,pHA,pWin)
    
    init_belief_pHA = 0.5
    a,d=get_T_maze_model(init_belief_pHA,pWin,100.0)
    # b = [0.1*np.ones_like(Bf)+Bf for Bf in B]
    b = [200*Bf for Bf in B]
    
    learn_dictionnary = get_learning_options(True,True,True,run_smoother=True)
    
    rngkey = jax.random.PRNGKey(26)
    Nos = [Ai.shape[0] for Ai in A]
    Np = U.shape[0]
    Ns = 8 # 4 x 2, should be automated
    source_space_shape = (4,2)   
    
    [all_obs_arr,all_true_s_arr,all_u_arr,all_qs_arr,all_qpi_arr,efes_arr,a_hist,b_hist,d_hist] = run_training_multi_factors(rngkey,Ns,Nos,Np,
            Ntrials,T,
            A,B,D,U,
            a,b,C,d,E,
            Th = Th,
            selection_method="stochastic",alpha = 16,gamma = None, 
            planning_options=DEFAULT_PLANNING_OPTIONS,
            learn_dictionnary=learn_dictionnary)
    
    
    trial_index = 0
    
    # To learn, we need : 
    # 1. an history of observations, vectorized : 
    obs_hist = [o_mod[trial_index] for o_mod in all_obs_arr]
    # print(obs_hist)
    
    # 2. an history of selected actions, vectorized : 
    u_hist = all_u_arr[trial_index]
    # print(u_hist)
    
    # 3. an history of state inferences, vectorized : 
    qs_hist = all_qs_arr[trial_index]
    
    # action_shapes = [B_f.shape[-1] for B_f in B]
    
    # Nf = len(B)

if __name__ == '__main__': 
    # Environment variables
    Nsubjects = 1
    Ntrials = 1
    T = 3
    Th = 2

    pContext = 0.0
    pHA = 1.0
    pWin = 0.98
    A,B,D,E,U = get_T_maze_gen_process(pContext,pHA,pWin)
    
    init_belief_pHA = 1.0
    initial_hint_confidence = 200
    la = -4
    rs = 2
    a,b,c,d =get_T_maze_model(init_belief_pHA,pWin,initial_hint_confidence,la,rs)
    
    learn_dictionnary = get_learning_options(True,False,True,run_smoother=True)
    
    rngkey = jax.random.PRNGKey(206)
    Nos = [Ai.shape[0] for Ai in A]
    Np = U.shape[0]
    Ns = 8 # 4 x 2, should be automated
    source_space_shape = (4,2)   
    
    diagnostic = run_training_multi_factors(rngkey,Ns,Nos,Np,
            Ntrials,T,
            A,B,D,U,
            a,b,c,d,E,
            Th = Th,
            selection_method="stochastic",alpha = 16,gamma = None, 
            planning_options=DEFAULT_PLANNING_OPTIONS,
            learn_dictionnary=learn_dictionnary) 
    
    
    exit()
    [all_obs_arr,all_true_s_arr,all_u_arr,all_qs_arr,all_qpi_arr,efes_arr,a_hist,b_hist,d_hist] = run_training_multi_factors(rngkey,Ns,Nos,Np,
            Ntrials,T,
            A,B,D,U,
            a,b,c,d,E,
            Th = Th,
            selection_method="stochastic",alpha = 16,gamma = None, 
            planning_options=DEFAULT_PLANNING_OPTIONS,
            learn_dictionnary=learn_dictionnary) 
    

    
    print(d_hist)
    print(np.round(np.array(all_qpi_arr),2))
    # for mod in range(2):
    #     print(np.round(np.array(a_hist[0][...,mod]),2))
    
    exit()
    from actynf.jax_methods.layer_learn import forwards_pass,backwards_pass,smooth_posterior_after_trial
    
    trial_idx = 0
    o = [o_m[trial_idx] for o_m in all_obs_arr]
    qs = all_qs_arr[trial_idx]
    act_h = all_u_arr[trial_idx]
    print()
    print(o)
    print(qs)
    print(act_h)
    
    print("-----------------------------")
    # ll_for,forwards_smooths = forwards_pass(
    #     o,qs,act_h,
    #     a,b,d,U
    # )
    # print(np.round(np.array(forwards_smooths),2))
    
    
    # print("----------------------------")
    # ll_back,backwards_smooths = backwards_pass(
    #     o,qs,act_h,
    #     a,b,d,U
    # )
    
    val = smooth_posterior_after_trial(o,qs,act_h,
        a,b,d,U,
        filter_type="two_filter")
    print(np.round(np.array(val),2))
    
    val = smooth_posterior_after_trial(o,qs,act_h,
        a,b,d,U,
        filter_type="one_filter")
    print(np.round(np.array(val),2))
    
    # print(np.round(np.array(backwards_smooths),2))
    # print("################################")
    # integrator = (lambda X : X/jnp.sum(X,axis=-1,keepdims=True))
    # smooth_posterior = integrator(forwards_smooths*backwards_smooths)
    # print(np.round(np.array(smooth_posterior),2))
    # print(ll_for)
    # print(ll_back)
    # print(smoothed_posteriors)
    # print(smoothed_posteriors.shape)
    
    # print(np.round(np.array(smoothed_posteriors),2))
    
    
    # ex_arr = jnp.array([
    #     [5,6],
    #     [9,7]
    # ])
    # print(forwards_smooths*backwards_smooths)
    # val,_ = _normalize(forwards_smooths*backwards_smooths,axis=-1)
    # print(np.round(np.array(val),2))