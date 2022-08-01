import numpy as np
import statistics as stat
import math
from ...model.metrics import flexible_entropy,flexible_kl_dir

from ...layer.layer_learn import MemoryDecayType
from ...base.miscellaneous_toolbox import isField
from ...base.function_toolbox import normalize
from ...base.matrix_functions import matrix_distance_list,argmean

from ...model.active_model import ActiveModel
from ...model.active_model_save_manager import ActiveSaveManager
from ...base.normal_distribution_matrix import generate_normal_dist_along_matrix,generate_normal_dist_along_mulist


def climb_stairs_B(pb=1,npoub=0):
    ns = 5
    B_mental_states = np.zeros((ns,ns,ns+npoub))

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

    for k in range(ns,ns+npoub):
        B_mental_states[:,:,k] = normalize(np.random.random((5,5))) # If we want a random matrix for neutral actions
                                                                    # Ill-advised for a climb stairs paradigm as random weightsq may provide a "shortcut"
        B_mental_states[:,:,k] = np.eye(5)
    return [B_mental_states]

def eye_skew_mean(skew,size):
    """ Useless :'( """
    true_A = np.zeros((size,size))
    j = size
    i = size
    for k in range(i):
        index = k + skew
        if (index<=-1):
            true_A[0,k] = 1
        elif (index>=j):
            true_A[-1,k]=1
        else :
            true_minor = math.floor(index)
            true_major = math.floor(index)+ 1
            minor = max(true_minor,0)
            major = min(true_major,j-1)
            distance_between_true_value_and_minor = abs(index-true_minor)
            distance_between_true_value_and_major = abs(index-true_major)
            low_value = 1-distance_between_true_value_and_minor # The further we are from the true value, the less the value (linear)
            high_value = 1-distance_between_true_value_and_major
            true_A[minor,k] = true_A[minor,k] + low_value
            true_A[major,k] = true_A[major,k] + high_value
    return true_A


def nf_model(modelname,savepath,prop_poubelle = 0.0,
                        prior_A_meanskew=0,prior_A_sigma = 1.0,prior_A_strength=1,
                        learn_a = True,prior_a_meanskew=0,prior_a_sigma = 1.0,prior_a_strength=1,
                        learn_b= True,
                        learn_d= True,
                        mem_dec_type=MemoryDecayType.NO_MEMORY_DECAY,mem_dec_halftime=5000,
                        perfect_a = False,perfect_A=False,verbose = False):
    """ 
    A is perfect, and the agent has flat priors regarding ACTION & PERCEPTION dynamics.
    We can choose to help him a little by providing it with indications.
    ACTION :
    The agent beliefs about state transitions are "plateau-like" :  
                                Prob density:       ^
                                                    |           ___
                                                    |          |   |        __
                                                    |__________|   |_______|  |______
                                           For a given initial state, prob to get to another          
    *strength* describes the strongness of the priors, how confident the agent is about those and how difficult it will be to change them
    *ratio* describes a preferential prior. If ratio > 1, the agent will have a positive prior regarding a certain dynamic
     e.g. if ratio =2, the agent will believe that action 2 at state 1 is twice as likely to lead to a hidden state of 2 than any other.
     [ this also stands for actions] 
    PERCEPTION :
    The agent beliefs about state-observations correspondance are "gaussian-like" :  
                                Prob density:       ^           Gaussian mean
                                                    |               ___|___
                                                    |       _______|       |______
                                                    |______|                       |__________
                                           For a given state, prob to get a given observation        
    Because there is a notion of continuity between observations, a gaussian prior seems adapted for PERCEPTION
    meaning that the agent may believe strongly that an observed 2 means a hidden state 2, but he may also believe
    less strongly that it may be 1 or 3, but it is much less likely to be 0 or 4.
    """
    constant = 20
    Nf = 1

    D_ = [np.array([1,1,0,0,0])] #All subjects start all trials either at state 0 or 1, forcing them to perform at least 3 sensible actions to get to the best state
    D_ = normalize(D_)

    d_ =[]
    #d_.append(np.array([0.996,0.001,0.001,0.001,0.001])) #[Terrible state, neutral state , good state, great state, excessive state]
    d_.append(np.zeros(D_[0].shape))

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # OBSERVATIONS : 
    # Generally : A[modality] is of shape (Number of outcomes for this modality) x (Number of states for 1st factor) x ... x (Number of states for nth factor)
    # prior_a_sigma : true values are the mean of our model
    # prior_strength : Base weight --> The higher this number, the stronger priors are and the longer it takes for experience to "drown" them \in [0,+OO[

    perfect_perception = np.eye(5)
    zeromatrix = np.zeros((5,5))
    if not(perfect_A):
        list_of_A_mean = []
        for k in range(5):
            list_of_A_mean.append(k+prior_A_meanskew)
        list_of_A_mean = np.array(list_of_A_mean)
        A_ = [prior_A_strength*generate_normal_dist_along_mulist(zeromatrix,list_of_A_mean,prior_A_sigma)]
    else :
        A_ = [np.eye(5)]
    
    if not(perfect_a):
        list_of_a_mean = []
        for k in range(5):
            list_of_a_mean.append(k+prior_a_meanskew)
        list_of_a_mean = np.array(list_of_a_mean)
        a_ = [prior_a_strength*generate_normal_dist_along_mulist(zeromatrix,list_of_a_mean,prior_a_sigma)]
    else :
        a_ = [np.eye(5)]
        learn_a = False

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # ACTIONS :
    # Transition matrixes between hidden states ( = control states)
    nu = 5
    npoubelle = int((prop_poubelle/(1-prop_poubelle))*nu) # Prop poubelle represents the proportion of mental actions with a neutral impact on the hidden states, rendering 
                                                          # them useless to improve one's mental state. Increasing this quantity is supposed to make exploration harder and 
                                                          # training longer.
    B_ = climb_stairs_B(pb=1,npoub=npoubelle)
    
    
    # Action model :    
    # Quality of the prior
    b_ = [np.ones(B_[0].shape)] # We start with no knowledge whatsoever about
                                # mental actions effects

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # PREFERENCES :
    # Transition matrixes between hidden states ( = control states)
    # For now, just a linear model, where dc/ds = cst
    la = -2
    rs = 2
    C_mental = np.array([[2*la],
                        [la],
                        [0],
                        [1*rs],
                        [2*rs]])
    C_ = [C_mental]
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # POLICIES
    Np = nu + npoubelle #Number of policies
    Nf = 1 #Number of state factors

    U_ = np.zeros((Np,Nf)).astype(int)
    U_[:,0] = range(Np)

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # HABITS
    # For now, no habits and we don't learn those. At some point, we will have to implement it
    E_ = None
    e_ = np.ones((Np,))


    T = 10
    savemanager = ActiveSaveManager(T,trial_savepattern=1,intermediate_savepattern=0,verbose=verbose,modelname=modelname,folder_name=savepath)
                                    # Trial related save , timestep related save
    nf_model = ActiveModel(savemanager)
    nf_model.T = T

    nf_model.A = A_
    nf_model.a = a_
    nf_model.layer_options.learn_a = learn_a

    nf_model.B = B_
    nf_model.b = b_
    nf_model.layer_options.learn_b = learn_b
    
    nf_model.D = D_
    nf_model.d = d_
    nf_model.layer_options.learn_d = learn_d

    nf_model.C = C_

    nf_model.U = U_

    nf_model.layer_options.T_horizon = 2
    nf_model.layer_options.learn_during_experience = False
    
    nf_model.layer_options.memory_decay = mem_dec_type
    nf_model.layer_options.decay_half_time = mem_dec_halftime

    nf_model.verbose = verbose

    return nf_model

def evaluate_container(container,options=['2','all']):
    """ Calculate non-array indicators to store in a pandas framework for further analysis and vizualization."""
    matrix_metric = options[0]
    dir_type = options[1]

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

    belief_about_states = container.X
    true_state_distribution = [np.zeros(belief_about_states[0].shape)]
       
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
        true_state_distribution[0][actual_state,t] = 1

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
        
    mean_errors_state = [mean_errors_state/T] # List because one for each layer factor
    mean_error_behaviour = [mean_error_behaviour/(T-1)] # List because one for each layer factor
    mean_error_observations = [mean_error_observations/T] # List because one for each layer modality
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

    # KL dirs w.r.t the true process matrices
    if (dir_type=='mean'):
        # Mean of all modalities / factors
        a_dir = stat.mean(flexible_kl_dir(normalize(container.a_),container.A_,option='centered'))
        b_dir = stat.mean(flexible_kl_dir(normalize(container.b_),container.B_,option='centered'))
        d_dir = stat.mean(flexible_kl_dir(normalize(container.d_),container.D_,option='centered'))
    else :
        # All modalities / factors : 
        a_dir = flexible_kl_dir(normalize(container.a_),container.A_,option='centered')
        b_dir = flexible_kl_dir(normalize(container.b_),container.B_,option='centered')
        d_dir = flexible_kl_dir(normalize(container.d_),container.D_,option='centered')

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