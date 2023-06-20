# This is empty for now, if we standardize model evaluation accross all 
# clim stairs paradigms, it may be interesting to define a single 
# evaluate container method like : 

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