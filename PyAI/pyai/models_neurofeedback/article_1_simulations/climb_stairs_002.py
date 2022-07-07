import numpy as np
import statistics as stat

from ...model.metrics import flexible_entropy,flexible_kl_dir

from ...layer.layer_learn import MemoryDecayType
from ...base.miscellaneous_toolbox import isField
from ...base.function_toolbox import normalize
from ...base.matrix_functions import matrix_distance_list,argmean

from ...model.active_model import ActiveModel
from ...model.active_model_save_manager import ActiveSaveManager
from ...base.normal_distribution_matrix import generate_normal_dist_along_matrix


def nf_model(modelname,savepath,prop_poubelle = 0.0,
                        learn_a = True,prior_a_sigma = 3,prior_a_strength=3,
                        learn_b=True,prior_b_sigma = 3,prior_b_strength=1,
                        learn_d=True,mem_dec_type=MemoryDecayType.NO_MEMORY_DECAY,mem_dec_halftime=5000,
                        verbose = False,constant = 20):   
    Nf = 1
    
    initial_state = 0
    D_ =[]
    D_.append(np.array([1,1,0,0,0])) #[Terrible state, neutral state , good state, great state, excessive state]
    #D_[0][initial_state] = 1
    D_ = normalize(D_)

    d_ =[]
    #d_.append(np.array([0.996,0.001,0.001,0.001,0.001])) #[Terrible state, neutral state , good state, great state, excessive state]
    d_.append(np.zeros(D_[0].shape))
    #d_ = D_


    # State Outcome mapping and beliefs
    # Prior probabilities about initial states in the generative process
    Ns = [5] #(Number of states)
    No = [5]

    # Observations : just the states 
    A_ = []

    # Generally : A[modality] is of shape (Number of outcomes for this modality) x (Number of states for 1st factor) x ... x (Number of states for nth factor)
    pa = 1
    A_obs_mental = np.array([[pa  ,0.5-0.5*pa,0         ,0         ,0   ],
                            [1-pa,pa        ,0.5-0.5*pa,0         ,0   ],
                            [0   ,0.5-0.5*pa,pa        ,0.5-0.5*pa,0   ],
                            [0   ,0         ,0.5-0.5*pa,pa        ,1-pa],
                            [0   ,0         ,0         ,0.5-0.5*pa,pa  ]])
    # A_obs_mental = np.array([[0,0,0,0,1],
    #                         [0,0,0,1,0],
    #                         [0,0,1,0,0],
    #                         [0,1,0,0,0],
    #                         [1,0,0,0,0]])
    A_ = [A_obs_mental]



    # prior_ratio = 5 # Correct_weights = ratio*incorrect_weights --> The higher this ratio, the better the quality of the priors
    # prior_strength = 10.0 # Base weight --> The higher this number, the stronger priors are and the longer it takes for experience to "drown" them \in [0,+OO[
    a_ = []
    a_.append(constant*prior_a_strength*generate_normal_dist_along_matrix(A_[0],prior_a_sigma)+1)

    temporary = a_
    A_ = a_
    a_ = temporary

    # Transition matrixes between hidden states ( = control states)
    pb = 1

    nu = 5
    npoubelle = int((prop_poubelle/(1-prop_poubelle))*nu)
    B_ = []
    B_mental_states = np.zeros((Ns[0],Ns[0],nu+npoubelle))

    # Line = where we're going
    # Column = where we're from
    B_mental_states[:,:,0] = np.array([ [1  ,1  ,1,1,1],         # Try to move to terrible state from others
                                        [0  ,0  ,0,0,0],
                                        [0  ,0  ,0,0,0],
                                        [0  ,0  ,0,0,0],
                                        [0  ,0  ,0,0,0]])

    B_mental_states[:,:,1] = np.array([[1-pb,0  ,0  ,0  ,0  ],         # Try to move to neutral state from others
                                        [pb ,1  ,1  ,1  ,1  ],
                                        [0  ,0  ,0  ,0  ,0  ],
                                        [0  ,0  ,0  ,0  ,0  ],
                                        [0  ,0  ,0  ,0  ,0 ]])

    B_mental_states[:,:,2] = np.array([ [1  ,0   ,0  ,0   ,0  ],         # Try to move to good state from others
                                        [0  ,1-pb,0  ,0   ,0  ],
                                        [0  ,pb  ,1  ,1   ,1  ],
                                        [0  ,0   ,0  ,0   ,0  ],
                                        [0  ,0   ,0  ,0   ,0  ]])

    B_mental_states[:,:,3] = np.array([ [1  ,0  ,0   ,0  ,0  ],         # Try to move to target state from others
                                        [0  ,1  ,0   ,0  ,0  ],
                                        [0  ,0  ,1-pb,0  ,0  ],
                                        [0  ,0  ,pb  ,1  ,1  ],
                                        [0  ,0  ,0   ,0  ,0  ]])

    B_mental_states[:,:,4] = np.array([ [1  ,0  ,0  ,0  ,1-pb],         # Try to move to best state from others
                                        [0  ,1  ,0  ,0  ,0  ],
                                        [0  ,0  ,1  ,0  ,0  ],
                                        [0  ,0  ,0  ,1-pb,0  ],
                                        [0  ,0  ,0  ,pb ,pb]])

    for k in range(nu,nu+npoubelle):
        B_mental_states[:,:,k] = normalize(np.random.random((5,5)))
        B_mental_states[:,:,k] = np.eye(5)

    B_.append(B_mental_states)

    b_ = []
    b_.append(constant*prior_b_strength*generate_normal_dist_along_matrix(B_[0],prior_b_sigma)+1)
    
    # print(b_)
    # print(b_[0].shape)
    la = -2
    rs = 2
    C_mental = np.array([[2*la],
                        [1*la],
                        [0],
                        [1*rs],
                        [2*rs]])
    C_ = [C_mental]

    NU = nu + npoubelle


    # Policies
    Np = NU #Number of policies
    Nf = 1 #Number of state factors

    U_ = np.zeros((NU,Nf)).astype(int)
    U_[:,0] = range(NU)

    #Habits
    E_ = None
    e_ = np.ones((Np,))


    T = 10
    savemanager = ActiveSaveManager(T,trial_savepattern=1,intermediate_savepattern=0,verbose=verbose,modelname=modelname,folder_name=savepath)
                                    # Trial related save , timestep related save
    nf_model = ActiveModel(savemanager)
    nf_model.T = T

    nf_model.A = A_
    nf_model.a = a_
    nf_model.B = B_
    nf_model.b = b_
    nf_model.C = C_
    nf_model.D = D_
    nf_model.d = d_
    nf_model.U = U_

    nf_model.layer_options.learn_a = learn_a
    nf_model.layer_options.learn_b = learn_b
    nf_model.layer_options.learn_d = learn_d

    nf_model.layer_options.T_horizon = 2
    nf_model.layer_options.learn_during_experience = False
    
    nf_model.layer_options.memory_decay = mem_dec_type
    nf_model.layer_options.decay_half_time = mem_dec_halftime

    nf_model.verbose = verbose

    return nf_model

def evaluate_container(container,options=['2']):
    """ Calculate non-array indicators to store in a pandas framework for further analysis and vizualization."""
    matrix_metric = options[0]

    trial = container.trial
    T = container.T
    Nf = len(container.s)

    def best_actions(actual_state):
        if(actual_state==0):
            return [1]
        if(actual_state==1):
            return [2]
        if(actual_state==2):
            return [3]
        if(actual_state==3):
            return [4]
        if(actual_state==4):
            return [4,5,6]
        
    # DIFFERENCE BETWEEN OPTIMAL STATES AND ACTUAL STATES DURING THE WHOLE TRIAL: 
    factor = 0
    modality = 0

    mean_errors_state = 0
    mean_error_behaviour = 0
    mean_error_observations = 0
    mean_error_percept = 0

    belief_about_states = container.X[factor]
    true_state_distribution = np.zeros(belief_about_states.shape)
       
    max_size = container.D_[factor].shape[0]- 1 # best possible factor
    
    init_actual_state = container.s[factor,0] 
    for t in range(T):
        # -------------prepare for state error-----------------
        optimal_state = min(init_actual_state+t,max_size) 
                    # Specific to action sequence learning problems : the optimal is the correct succession of states up to the best state ( 0 -> 1 -> ... -> max_size)
        actual_state = container.s[factor,t] 
        if(optimal_state==0):
            mean_errors_state += 0
        else :
            mean_errors_state += abs(optimal_state-actual_state)/optimal_state

        # -------------prepare for perception error-----------------
        true_state_distribution[actual_state,t] = 1

        # -------------prepare for observations error-----------------
        optimal_observation = min(init_actual_state+t,max_size) 
        actual_observation = container.o[modality,t]
        if(optimal_observation==0):
            mean_error_observations += 0
        else :
            mean_error_observations += abs(optimal_observation-actual_observation)/optimal_observation

        # -------------prepare for action error-----------------
        if (t<T-1):
            optimal_action = best_actions(actual_state)
                        # Specific to action sequence learning problems : the optimal action is the correct succession of actions up to the best state 
            actual_action = container.u[factor,t] 
            #print(optimal_action,actual_action,actual_state)
            if not(actual_action in optimal_action) :
                mean_error_behaviour += 1 # Binary (best action chosen ? y/n)
        
    mean_errors_state = mean_errors_state/T
    mean_error_behaviour = mean_error_behaviour/(T-1)
    mean_error_observations = mean_error_observations/T
    mean_error_percept = flexible_kl_dir(belief_about_states,true_state_distribution,option='centered')


    # Matrix distances (not that useful ?)
    A_mean_distance = stat.mean(matrix_distance_list(container.A_,normalize(container.a_),metric=matrix_metric))
    B_mean_distance = stat.mean(matrix_distance_list(container.B_,normalize(container.b_),metric=matrix_metric))
    C_mean_distance = stat.mean(matrix_distance_list(container.C_,normalize(container.c_),metric=matrix_metric))
    D_mean_distance = stat.mean(matrix_distance_list(container.D_,normalize(container.d_),metric=matrix_metric))
    if (isField(container.E_)):
        E_mean_distance = stat.mean(matrix_distance_list(container.E_,normalize(container.e_),metric=matrix_metric))
    else :
        E_mean_distance = 0
    # We can give a simple normalization relying on the fact that for normalized distribution matrices,
    # d_2(a,b) <= sqrt(2*number_of_columns) [In practice, d_2(a,b) < sqrt(2*number_of_columns)/2]    
    
    # KL dirs (calculated in the respective free energies :D) of matrices compared to their prior values (same trial)
    free_energy_a = container.FE['Fa']
    free_energy_b = container.FE['Fb']
    free_energy_c = container.FE['Fc']
    free_energy_d = container.FE['Fd']
    free_energy_e = container.FE['Fe']

    # KL dirs w.r.t the true process matrices : 
    # print(normalize(container.a_))
    # print(container.A_)
    # print(flexible_kl_dir(container.a_,container.A_,option='centered'))
    a_dir = stat.mean(flexible_kl_dir(normalize(container.a_),container.A_,option='centered'))
    b_dir = stat.mean(flexible_kl_dir(normalize(container.b_),container.B_,option='centered'))
    d_dir = stat.mean(flexible_kl_dir(normalize(container.d_),container.D_,option='centered'))

    #print(free_energy_a,free_energy_b,free_energy_c,free_energy_d,free_energy_e)
    
    factor = 0.5
    try :
        #mean_uncertainty_a = mean_uncertainty(container.a_,factor)
        mean_uncertainty_a = flexible_entropy(container.a_)
    except :
        mean_uncertainty_a = [0 for i in range(len(container.A_))]
    try :
        #mean_uncertainty_b = mean_uncertainty(container.b_,factor)
        mean_uncertainty_b = flexible_entropy(container.b_)
    except :
        mean_uncertainty_b = [0 for i in range(len(container.B_))]
    try :
        #mean_uncertainty_d = mean_uncertainty(container.d_,factor)
        mean_uncertainty_d = flexible_entropy(container.d_)
    except :
        mean_uncertainty_d = [0 for i in range(len(container.D_))]

    output_dict = {
        'mean_error_perception' : mean_error_percept,
        'mean_error_state':mean_errors_state, # Global error cmpred to global optimal succession of states
        'mean_error_behaviour':mean_error_behaviour, # Local error cmpred to optimal action
        'mean_error_observations':mean_error_observations,
        'fe_a':free_energy_a,
        'fe_b':free_energy_b,
        'fe_c':free_energy_c,
        'fe_d':free_energy_d,
        'fe_e':free_energy_e,
        'a_dist':A_mean_distance,
        'b_dist':B_mean_distance,
        'c_dist':C_mean_distance,
        'd_dist':D_mean_distance,
        'e_dist':E_mean_distance,
        'a_dir':a_dir,
        'b_dir':b_dir,
        'd_dir':d_dir,
        'a_uncertainty': mean_uncertainty_a,
        'b_uncertainty': mean_uncertainty_b,
        'd_uncertainty': mean_uncertainty_d
    }
    return output_dict    
