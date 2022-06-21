from pickle import FALSE
import numpy as np
import os
import PIL
import matplotlib.pyplot as plt
import statistics as stat
from pyai.base.file_toolbox import save_flexible,load_flexible
from pyai.model.metrics import flexible_entropy,flexible_kl_dir

from pyai.layer.layer_learn import MemoryDecayType
from pyai.base.miscellaneous_toolbox import isField
from pyai.base.function_toolbox import normalize
from pyai.base.matrix_functions import matrix_distance_list,argmean

from pyai.model.active_model import ActiveModel
from pyai.model.active_model_save_manager import ActiveSaveManager
from pyai.model.active_model_container import ActiveModelSaveContainer

from pyai.model.model_visualizer import belief_matrices_plots,generate_model_sumup,general_performance_plot,trial_plot_figure

def nf_model(modelname,savepath,prop_poubelle = 0.0,
                        learn_a = True,prior_a_ratio = 3,prior_a_strength=3,
                        learn_b=True,prior_b_ratio = 0.0,prior_b_strength=1,
                        learn_d=True,mem_dec_type=MemoryDecayType.NO_MEMORY_DECAY,mem_dec_halftime=5000):
    
    def base_prior_generator(true_matrix,strength_of_false,strength_of_true,eps=1e-8):
        prior = np.zeros(true_matrix.shape)
        ones = np.ones(true_matrix.shape)

        false_vals_mask = true_matrix<eps
        prior[false_vals_mask] = prior[false_vals_mask] + strength_of_false*ones[false_vals_mask]

        true_vals_mask = true_matrix>=eps
        prior[true_vals_mask] = prior[true_vals_mask] + strength_of_true*ones[true_vals_mask]

        return prior
    
    def gaussian_prior_generator(true_matrix,sigma,eps=1e-8):
        return None # TBI
    
    
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
    a_.append(np.ones((A_[0].shape))*prior_a_strength)
    a_[0] = a_[0] + (prior_a_ratio-1.0)*prior_a_strength*A_[0]

    a_[0] = np.ones(A_[0].shape)*prior_a_strength + (prior_a_ratio-1.0)*prior_a_strength*np.eye(A_[0].shape[0])
    #a_[0] = np.eye(5)


    # Transition matrixes between hidden states ( = control states)
    pb = 1

    nu = 5
    prop_poublle = 0.3
    npoubelle = int((prop_poublle/(1-prop_poublle))*nu)
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

    b_ = [np.ones((B_[0].shape))]
    # b_[0][0,:,:] = 0.1
    # b_[0][1,:,:] = 0.15
    # b_[0][2,:,:] = 0.2
    # b_[0][3,:,:] = 0.25
    # b_[0][4,:,:] = 0.3

    #b_[0] = 1.0*b_[0] - 0.0*B_[0]
    
    b_ = []
    b_.append(np.ones((B_[0].shape))*prior_b_strength)
    b_[0] = b_[0] + (prior_b_ratio-1.0)*prior_b_strength*B_[0]


    #b_ = B_
    # for i in range(B_[0].shape[-1]):
    #     b_[0][:,:,i][B_[0][:,:,i]>0.5] += 10

    No = [A_[0].shape[0]]

    la = -2
    rs = 2
    C_mental = np.array([[2*la],
                        [la],
                        [rs],
                        [3*rs],
                        [14*rs]])
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
    savemanager = ActiveSaveManager(T,trial_savepattern=1,intermediate_savepattern=0)
                                    # Trial related save , timestep related save
    nf_model = ActiveModel(savemanager,modelname,savepath)
    nf_model.T = T
    nf_model.A = A_
    nf_model.a = a_
    #nf_model.a = A_
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

    return nf_model

def evaluate_container(container,matrix_metric='2'):
    """ Calculate non-array indicators to store in a pandas framework for further analysis and vizualization."""
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
    mean_errors_state = 0
    mean_error_behaviour = 0
       
    max_size = container.D_[factor].shape[0]- 1 # best possible factor
    
    init_actual_state = container.s[factor,0] 
    for t in range(T):
        optimal_state = min(init_actual_state+t,max_size) 
                    # Specific to action sequence learning problems : the optimal is the correct succession of states up to the best state ( 0 -> 1 -> ... -> max_size)
        actual_state = container.s[factor,t] 
        if(optimal_state==0):
            mean_errors_state += 0
        else :
            mean_errors_state += abs(optimal_state-actual_state)/optimal_state
        
        if (t<T-1):
            optimal_action = best_actions(actual_state)
                        # Specific to action sequence learning problems : the optimal action is the correct succession of actions up to the best state 
            actual_action = container.u[factor,t] 
            #print(optimal_action,actual_action,actual_state)
            if not(actual_action in optimal_action) :
                mean_error_behaviour += 1 # Binary (best action chosen ? y/n)
        
    mean_errors_state = mean_errors_state/T
    mean_error_behaviour = mean_error_behaviour/(T-1)
    
    
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

    mean_error_percept = 0
    output_dict = {
        'mean_perception_error' : mean_error_percept,
        'mean_error_state':mean_errors_state, # Global error cmpred to global optimal succession of states
        'mean_error_behaviour':mean_error_behaviour, # Local error cmpred to optimal action
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



def general_performance_indicators(savepath,modelname,instance_number=0):
    def number_of_trials_in_instance_folder(path):
        counter = 0
        for file in os.listdir(path):
            instance = int(file.split("_")[0])
            counter = counter + 1
        return counter

    trials = []
    Ka = []
    Kb = []
    Kd = []
    a_err = []
    b_err = []
    error_states = []
    error_behaviour = []
    instance_string = f'{instance_number:03d}'
    instance_folder = os.path.join(savepath,modelname,instance_string)
    total_trials = number_of_trials_in_instance_folder(instance_folder)
    for trial in range(total_trials):
        cont = ActiveSaveManager.open_trial_container(os.path.join(savepath,modelname),instance_number,trial,'f')
        eval_cont = evaluate_container(cont)
        trials.append(trial) 
        a_err.append(eval_cont['a_dir'])
        b_err.append(eval_cont['b_dir'])
        Ka = Ka + eval_cont['a_uncertainty']
        Kb = Kb + eval_cont['b_uncertainty']
        Kd = Kd + eval_cont['d_uncertainty']

        error_states.append(eval_cont['mean_error_state'])
        error_behaviour.append(eval_cont['mean_error_behaviour'])
    return trials,a_err,b_err,Ka,Kb,Kd,error_states,error_behaviour
 
def all_indicators(modelname,savepath) : 
    """Return all the performance indicators implemented for a given model accross all layer instances
    TODO : reimplement using the "general performance indicators" function !"""
    
    loadpath = os.path.join(savepath,modelname)

    A_list = []
    B_list = []
    D_list = []
    
    Ka = []
    Kb = []
    Kd = []
    a_err = []
    b_err = []
    error_states = []
    error_behaviour = []         
    model = ActiveModel.load_model(loadpath)

    for file in os.listdir(loadpath):
        complete_path = os.path.join(loadpath,file)
        is_file = (os.path.isfile(complete_path))
        is_dir = (os.path.isdir(complete_path))

        if (is_dir) :
            if ("_RESULTS" in file) or ("_MODEL" in file):
                #ignore this file 
                print("Ignoring file " + file)
                continue
            print("Adding file  : "+ file + " to the mean trial.")
            A_list.append([])
            B_list.append([])
            D_list.append([])

            Ka.append([])
            Kb.append([])
            Kd.append([])
            a_err.append([])
            b_err.append([])
            error_states.append([])
            error_behaviour.append([])

            # This is trial results (layer instance)
            layer_instance = int(file)


            len_dir = len(os.listdir(complete_path)) # All the trials

            for newfile in os.listdir(complete_path): # Newfile is the sumup for all trials
                L = newfile.split("_")
                if (L[-1] != 'f') :
                    continue
                
                trial_counter = int(L[0])
                timestep_counter = 'f'
                cont = ActiveSaveManager.open_trial_container(loadpath,layer_instance,trial_counter,timestep_counter)
                
                
                # A is matrix of dimensions outcomes x num_factors_for_state_factor_0 x num_factors_for_state_factor_1 x ... x num_factors_for_state_factor_n
                # draw a 3D image is not made for matrix with dimensions >= 4. As a general rule, we take the initial dimensions of the matrix and pick the 0th 
                # Indices of excess dimensions :
                    
                try :
                    a_mat = cont.a_
                except :
                    a_mat = cont.A_
                A_list[-1].append(a_mat)

                try :
                    b_mat = cont.b_
                except :
                    b_mat = cont.B_
                B_list[-1].append(b_mat)

                try :
                    d_mat = cont.d_
                except :
                    d_mat = cont.D_
                D_list[-1].append(d_mat)

                eval_cont = evaluate_container(cont)
                a_err[-1].append(eval_cont['a_dir'])
                b_err[-1].append(eval_cont['b_dir'])

                Ka[-1] = Ka[-1] + eval_cont['a_uncertainty']
                Kb[-1] = Kb[-1] + eval_cont['b_uncertainty']
                Kd[-1] = Kd[-1] + eval_cont['d_uncertainty']
                error_states[-1].append(eval_cont['mean_error_state'])
                error_behaviour[-1].append(eval_cont['mean_error_behaviour'])
    Ka_arr = np.array(Ka)
    Kb_arr = np.array(Kb)
    Kd_arr = np.array(Kd)
    a_err_arr = np.array(a_err)
    b_err_arr = np.array(b_err)
    error_states_arr = np.array(error_states)
    error_behaviour_arr = np.array(error_behaviour)
    return A_list,B_list,D_list,Ka_arr,Kb_arr,Kd_arr,a_err_arr,b_err_arr,error_states_arr,error_behaviour_arr

def mean_indicators(A_list,B_list,D_list,Ka_arr,Kb_arr,Kd_arr,a_err_arr,b_err_arr,error_states_arr,error_behaviour_arr):
    def mean_over_first_dim(list_of_list_of_matrices):
        def flexible_sum(list_of_matrices_1,list_of_matrices_2):
            assert len(list_of_matrices_1)==len(list_of_matrices_2),"List should be equal dimensions before summing"
            r = []
            for k in range(len(list_of_matrices_1)) :
                r.append(list_of_matrices_1[k] + list_of_matrices_2[k])
            return r
        
        r = [0 for i in range(len(list_of_list_of_matrices[0]))]
        cnt = 0
        for list_of_matrices in list_of_list_of_matrices :
            r = flexible_sum(r, list_of_matrices)
            cnt = cnt + 1.0

        # Mean :
        for k in range(len(list_of_list_of_matrices[0])):
            r[k] = r[k]/cnt
        
        return r

    mean_A = []
    mean_B = []
    mean_D = []
    total_instances = len(A_list)
    for t in range(len(A_list[0])): # Iterating through timesteps
        a_at_t = []
        b_at_t = []
        d_at_t = []
        for k in range(len(A_list)):
            a_at_t.append(normalize(A_list[k][t]))
            b_at_t.append(normalize(B_list[k][t]))
            d_at_t.append(normalize(D_list[k][t]))

        mean_A.append(mean_over_first_dim(a_at_t))
        mean_B.append(mean_over_first_dim(b_at_t))
        mean_D.append(mean_over_first_dim(d_at_t))

    Ka_arr = np.mean(Ka_arr,axis=0)
    Kb_arr = np.mean(Kb_arr,axis=0)
    Kd_arr = np.mean(Kd_arr,axis=0)
    a_err_arr = np.mean(a_err_arr,axis=0)
    b_err_arr = np.mean(b_err_arr,axis=0)
    error_states_arr = np.mean(error_states_arr,axis=0)
    error_behaviour_arr = np.mean(error_behaviour_arr,axis=0)

    return mean_A,mean_B,mean_D,a_err_arr,b_err_arr,Ka_arr,Kb_arr,Kd_arr,error_states_arr,error_behaviour_arr,total_instances

def mean_indicators_model(modelname,savepath) :
    """Generate the mean trial by selecting the mean value accross all instances for every matrix and error estimators    """
    A_list,B_list,D_list,Ka_arr,Kb_arr,Kd_arr,a_err_arr,b_err_arr,error_states_arr,error_behaviour_arr = all_indicators(modelname,savepath)
    return mean_indicators(A_list,B_list,D_list,Ka_arr,Kb_arr,Kd_arr,a_err_arr,b_err_arr,error_states_arr,error_behaviour_arr)



def generate_instances_figures(savepath,modelname,instance_list,gifs=False,mod_ind=0,fac_ind=0):
    generate_model_sumup(modelname,savepath,gifs,mod_ind,fac_ind)
    for inst in instance_list:
        general_performance_figure(savepath,modelname,inst)

def general_performance_figure(savepath,modelname,instance_number=0) :
    trials,a_err,b_err,Ka,Kb,Kd,error_states,error_behaviour = general_performance_indicators(savepath,modelname,instance_number)
    save_string = f'{instance_number:03d}'
    figtitle = modelname +" - Instance " + str(instance_number) + " performance sumup"
    general_performance_plot(savepath,modelname,save_string,trials,a_err,b_err,Ka,Kb,error_states,error_behaviour,smooth_window = 5,show=False,figtitle=figtitle)

def generate_mean_behaviour_figures(savepath,modelname,show=True):
    mean_A,mean_B,mean_D,a_err,b_err,Ka_arr,Kb_arr,Kd_arr,error_states_arr,error_behaviour_arr,tot_instances = mean_indicators_model(modelname,savepath)
    n = a_err.shape[0]
    trials = np.linspace(0,n,n)
    general_performance_figure(savepath,modelname,"GLOBAL",trials,a_err,b_err,Ka_arr,Kb_arr,error_states_arr,error_behaviour_arr,smooth_window=5,figtitle=modelname+" - performance sumup over " + str(tot_instances) + " instance(s)",show=True)
    belief_matrices_plots(modelname,savepath,mean_A,mean_B,mean_D,plot_gifs=True)


def trial_plot(plotfile,plotmean=False,action_labels="alphabet",title=None):
    hidden_state_factor = 0
    perc_modality = 0

    cont = ActiveModelSaveContainer.load_active_model_container(plotfile)
    eval_cont = evaluate_container(cont)

    T = cont.T
    
    obs = cont.o[perc_modality,:]
    states = cont.s[hidden_state_factor,:]
    acts = cont.u[hidden_state_factor,:]
    beliefs = cont.X[hidden_state_factor]
    u_post = cont.U_post

    # BEGIN ! --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
    try :
        a_mat = cont.a_[perc_modality]
    except:
        a_mat = cont.A_[perc_modality]
    while (a_mat.ndim < 3):
        a_mat = np.expand_dims(a_mat,-1)
    

    try :
        b_mat = cont.b_[hidden_state_factor]
    except :
        b_mat = cont.B_[hidden_state_factor]
    

    figure = trial_plot_figure(T,beliefs,u_post,
                obs,states,acts,
                a_mat,b_mat,
                plotmean=plotmean,action_labels=action_labels,title=title)
    figure.show()
    

def run_a_trial():
    # ENVIRONMENT
    savepath = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","code","results","lessgo")
    modelname = "initial_results_test_10a_4isbest"

    # SIMULATING TRAINING
    model = nf_model(modelname,savepath,prop_poubelle=0.3,prior_a_ratio=10,prior_a_strength=2,prior_b_ratio=1,prior_b_strength=1)
    Ninstances = 10
    trials_per_instances = 250
    model.initialize_n_layers(Ninstances)
    overwrite = False
    model.run_n_trials(trials_per_instances,overwrite=overwrite)

    # FIGURES AND ANALYSIS
    instance_list = [i for i in range(Ninstances)]
    modality_indice = 0
    factor_indice = 0
    gifs=True
    generate_instances_figures(savepath,modelname,instance_list,gifs=gifs,mod_ind=modality_indice,fac_ind=factor_indice)

    generate_mean_behaviour_figures(savepath,modelname,show=True)


    # DISPLAY TRIALS 
    model_folder = os.path.join(savepath,modelname)
    for instance in range(Ninstances) :
        for trial in [trials_per_instances-1] :
            full_file_name = ActiveSaveManager.generate_save_name(model_folder,instance,trial,'f')
            trial_plot(full_file_name,title="Trial " + str(trial) + " sum-up (instance " + str(instance) + " )")
    input()

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def run_models(models_dictionnary,Ntrials,Ninstances,overwrite = False):
    max_n = len(models_dictionnary)
    cnter = 0.0
    for key in models_dictionnary: 
        print("MODEL : " + key)
        model_options = models_dictionnary[key]
        a_learn = model_options[0]
        a_acc = model_options[1]
        a_str = model_options[2]
        b_learn = model_options[3]
        b_acc = model_options[4]
        b_str = model_options[5]
        d_learn = model_options[6]
        memory_decay_type = model_options[7]
        memory_decay_halftime = model_options[8]

        modelname = key

        # SIMULATING TRAINING
        model = nf_model(modelname,savepath,prop_poubelle=0.0,prior_a_ratio=a_acc,prior_a_strength=a_str,learn_a=a_learn,
                                                            prior_b_ratio=b_acc,prior_b_strength=b_str,learn_b=b_learn,
                                                            learn_d=d_learn,
                                                            mem_dec_type=memory_decay_type,mem_dec_halftime=memory_decay_halftime)
        model.initialize_n_layers(Ninstances)
        trial_times = [0.01]
        model.run_n_trials(Ntrials,overwrite=overwrite,global_prop = [cnter,max_n],list_of_last_n_trial_times=trial_times)
        cnter = cnter + 1

def sliding_window_mean(list_input,window_size = 5):
        list_output = []
        N = len(list_input)
        for trial in range(N):
            mean_value = 0
            counter = 0
            for k in range(trial - window_size,trial + window_size + 1):
                if(k>=0):
                    try :
                        mean_value += list_input[k]
                        counter += 1
                    except :
                        a = 0
                        #Nothing lol
            list_output.append(mean_value/counter)
        return list_output



savepath = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","code","results","series","series_a_b_prior")
models_dictionnary = {
    "a_ac1p5_str1_b_ac1_str1":[True,1.5,1,True,1,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000],
    "a_ac3_str1_b_ac1_str1":[True,3,1,True,1,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000],
    "a_ac5_str1_b_ac1_str1":[True,5,1,True,1,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000],
    "a_ac10_str1_b_ac1_str1":[True,10,1,True,1,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000],
    "a_ac15_str1_b_ac1_str1":[True,15,1,True,1,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000],
    "a_ac25_str1_b_ac1_str1":[True,25,1,True,1,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000],
    "a_ac50_str1_b_ac1_str1":[True,50,1,True,1,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000],
    "a_ac200_str1_b_ac1_str1":[True,200,1,True,1,1,True,MemoryDecayType.NO_MEMORY_DECAY,2000]
}

def generate_a_dictionnary(a_priors,b_priors) :
    new_dict = {}
    for ka in range(a_priors.shape[0]):
        for kb in range(b_priors.shape[0]):
            modelchar = [True,a_priors[ka],1,True,b_priors[kb],1,True,MemoryDecayType.NO_MEMORY_DECAY,2000]
            modelname = "a_ac"+str(int(10*a_priors[ka]))+"_str1_b_ac"+str(int(10*b_priors[kb]))+"_str1"
            new_dict[modelname] = modelchar
    return new_dict



prior_value_a = np.array([1.0,1.2,1.5,1.8,2.0,2.4,2.8,3.0,5.0,15.0,50.0,200.0])
#prior_value_a = np.array([1.0,1.2,1.5,2.0,5.0,15.0])
models_dictionnary = (generate_a_dictionnary(prior_value_a,prior_value_a))
Ninstances = 10
Ntrials = 500
overwrite = False
run_models(models_dictionnary,Ntrials,Ninstances,overwrite=overwrite)

# Multimodel plot :
all_beh_err = []
all_stat_err = []
all_a_err = []
all_b_err = []

# cnt = 0
# for key in models_dictionnary:
#     print(key)
#     mean_A,mean_B,mean_D,a_err_arr,b_err_arr,Ka_arr,Kb_arr,Kd_arr,error_states_arr,error_behaviour_arr,total_instances = mean_indicators_model(key,savepath)
#     all_beh_err.append(error_behaviour_arr)
#     all_stat_err.append(error_states_arr)
#     all_a_err.append(a_err_arr)
#     all_b_err.append(b_err_arr)

t = np.arange(0,Ntrials,1)
# arr  = (np.array(all_stat_err))

savenam = os.path.join(savepath,"output_array.my_arr")
# save_flexible(arr,savenam)

arr = load_flexible(savenam)
# print(arr.shape)
# for i in range(len(models_dictionnary)):
#     y = arr[i,:]
#     if y.shape[0]>Ntrials:
#         y = y[:Ntrials]
#     plt.scatter(t,y,s=1)

# # Single value of a or b

# for i in range(len(models_dictionnary)):
#     y = arr[i,:]
#     t = np.arange(0,Ntrials,1)
#     y_av = sliding_window_mean(list(y),4)
#     y_av = np.array(y_av)
#     if y_av.shape[0]>Ntrials:
#         y_av = y_av[:Ntrials]
    
#     my_key = list(models_dictionnary)[i]
#     list_of_key =  (my_key.split("_"))
#     a_acc = float(list_of_key[1].strip("ac"))/10
#     b_acc = float(list_of_key[4].strip("ac"))/10
#     print(a_acc,b_acc)
#     prior_value = models_dictionnary[list(models_dictionnary)[i]][1]
#     if (a_acc == 1.0) or (a_acc==1.5):
#         plt.plot(t,y_av,label="Good Prior Biais =  " + str(prior_value) + " b_acc = " + str(b_acc))

# plt.legend()
# plt.xlabel("Trials")
# plt.ylabel("State error w.r.t optimal")
# plt.title("How does prior influence overall performance")
# plt.grid(True)
# plt.show()

# 3D plot


the_t = 100
xs = []
ys = []
zs = []

icnt = 0
jcnt = 0
J = prior_value_a.shape[0]
plot_this = np.zeros((J,J))

for i in range(len(models_dictionnary)):
    y = arr[i,:]
    t = np.arange(0,Ntrials,1)
    y_av = sliding_window_mean(list(y),10)
    y_av = np.array(y_av)
    if y_av.shape[0]>Ntrials:
        y_av = y_av[:Ntrials]
    
    my_key = list(models_dictionnary)[i]
    list_of_key =  (my_key.split("_"))
    a_acc = float(list_of_key[1].strip("ac"))/10
    b_acc = float(list_of_key[4].strip("ac"))/10
    #if (a_acc < 50)and(b_acc<50):
    xs.append(a_acc)
    ys.append(b_acc)
    zs.append(y_av[the_t])
    plot_this[icnt,jcnt] = y_av[the_t]

    icnt = icnt+1
    if (icnt>=J):
        icnt = 0
        jcnt = jcnt+1



from matplotlib import cm
print(plot_this)
zs = np.array(zs)
X,Y = np.meshgrid(prior_value_a,prior_value_a)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs, ys, zs, cmap='Greens')
ax.plot_surface(X,Y,plot_this,linewidth=0,cmap=cm.coolwarm, antialiased=False)
plt.show()