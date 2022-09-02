import numpy as np
import statistics as stat

from ....model.metrics import flexible_entropy,flexible_kl_dir

from ....layer.layer_learn import MemoryDecayType
from ....base.miscellaneous_toolbox import isField
from ....base.function_toolbox import normalize
from ....base.matrix_functions import matrix_distance_list,argmean

from ....model.active_model import ActiveModel
from ....model.active_model_save_manager import ActiveSaveManager
from ....base.normal_distribution_matrix import generate_normal_dist_along_matrix


def alpha_assymetry_model(modelname,savepath,
            neurofeedback_training_group = 'right',
            learn_a = True,prior_a_precision = 1.0,prior_a_confidence=1,
            learn_b=True,prior_b_precision = 1.0,prior_b_confidence=1,
            learn_d=True,
            mem_dec_type=MemoryDecayType.NO_MEMORY_DECAY,mem_dec_halftime=5000,
            perfect_a = False,perfect_b=False,perfect_d = False,
            verbose = False,SHAM="False"):
    Nf = 1 # two mental states are interesting in this scenario
    Ns = [7]
    D_.append(np.array([0,0,0,1,0,0,0])) # Initial state (attention isn't focused either right or left)
    D_ = normalize(D_)

    d_ =[]
    if (perfect_d):
        d_ = D_
        learn_d = False
    else :
        d_.append(np.ones(D_[0].shape))
    # Neutral priors about the starting states

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # OBSERVATIONS : 
    # Generally : A[modality] is of shape (Number of outcomes for this modality) x (Number of states for 1st factor) x ... x (Number of states for nth factor)
    # Depending on the LNT or RNT, the A matrix is different :
    
    # Here, let's pick 5 feedback levels possible, equivalent to the
    # difference between the two states
    No = 7 # 1-very_bad 2-bad 3-neutral 4-good 5-very_good
    if SHAM=="False" :
        A_ = np.zeros((No,7))

        # Right is at zero :
        if (neurofeedback_training_group =="right"):
            # Left alpha is to be very distinct from right alpha
            # We model this with a simple state correlated to the level 
            # of alpha :
            # If right is low :
            A_[:,:] = np.array([[1,0,0,0,0,0,0],  # Very bad if attentive state = 0 [left]
                                [0,1,0,0,0,0,0],  # 
                                [0,0,1,0,0,0,0],  # 
                                [0,0,0,1,0,0,0],  # 
                                [0,0,0,0,1,0,0],
                                [0,0,0,0,0,1,0],
                                [0,0,0,0,0,0,1]]) # Very good if attentive state = max [right]
            
        elif (neurofeedback_training_group =="left"):
            # Left alpha is to be very distinct from right alpha
            # We model this with a simple state correlated to the level 
            # of alpha :
            # If left is low :
            A_[:,:] = np.array([[0,0,0,0,0,0,1],   # Very bad if attentive state = max [right]
                                [0,0,0,0,0,1,0],  
                                [0,0,0,0,1,0,0],
                                [0,0,0,1,0,0,0], 
                                [0,0,1,0,0,0,0],
                                [0,1,0,0,0,0,0],
                                [1,0,0,0,0,0,0]]) # Very good if attentive state = 0 [left]
        A_ = [A_]
    else :
        A_ = [normalize((No,)+tuple(Ns))]
    
    # prior_a_sigma : true values are the mean of our model
    # prior_strength : Base weight --> The higher this number, the stronger priors are and the longer it takes for experience to "drown" them \in [0,+OO[
    
    if (not(perfect_a)):
        # a priors are flat to begin with ?
        a_ = [0]
        a_[0] = prior_a_confidence*(np.ones(A_[0].shape) + (1-prior_a_precision)*A_[0])
    else : 
        a_ = A_
        learn_a=False
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # ACTIONS :
    # Transition matrixes between hidden states ( = control states)
    # B_ = climb_stairs_B(pb=1,npoub=npoubelle)
    
    to_the_right =np.array([[0,0,0,0,0,0,0],   # Very bad if attentive state = max [right]
                                 [1,0,0,0,0,0,0],  
                                 [0,1,0,0,0,0,0],
                                 [0,0,1,0,0,0,0], 
                                 [0,0,0,1,0,0,0],
                                 [0,0,0,0,1,0,0],
                                 [0,0,0,0,0,1,1]]) # Very good if attentive state = 0 [left]
    to_the_left = np.array([[1,1,0,0,0,0,0],   # Very bad if attentive state = max [right]
                            [0,0,1,0,0,0,0],  
                            [0,0,0,1,0,0,0],
                            [0,0,0,0,1,0,0], 
                            [0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,1],
                            [0,0,0,0,0,0,0]]) # Very good if attentive state = 0 [left]
    neutral_activity = np.eye(Ns[0])
    
    n_b = 6
    B_ = [np.zeros(tuple(Ns)+(n_b,)),np.zeros(tuple(Ns)+(n_b,))]

    for factor in range(Nf):
        B_[factor][:,:,0] = to_the_right
        B_[factor][:,:,1] = to_the_left
        for k in range(2,n_b):
            B_[factor][:,:,k] = neutral_activity
    
    if (perfect_b):
        b_ = B_
        learn_b = False
    else :
        b_ = [0,0]
        b_[0] = prior_b_confidence*(np.ones(B_[0].shape) + (1-prior_b_precision)*B_[0])
        b_[1] = prior_b_confidence*(np.ones(B_[1].shape) + (1-prior_b_precision)*B_[1])

    # U_ = np.zeros((nu,len(Ns)))
    U_ = np.expand_dims(np.array(range(n_b)),0)
    U_ = U_.astype(np.int)

    la = -2
    rs = 2
    C_mental = np.array([[2*la],
                        [la],
                        [0],
                        [1*rs],
                        [2*rs]])
    C_ = [C_mental] 

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # HABITS
    # For now, no habits and we don't learn those. At some point, we will have to implement it
    E_ = None
    e_ = np.ones((n_b,))


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