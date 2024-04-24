# -*- coding: utf-8 -*-
import sys,os

from cmath import pi
from turtle import width
from imageio import save
import numpy as np
import matplotlib.pyplot as plt

from pyai.base.function_toolbox import *
from pyai.base.miscellaneous_toolbox import flexible_copy

from pyai.model import active_model,active_model_container,active_model_save_manager
from pyai.neurofeedback_run import save_model_performance_dictionnary,load_model_performance_dictionnary
from pyai.models_neurofeedback.article_1_simulations.climb_stairs_flat_priors import nf_model,nf_model_imp2,nf_model_imp,evaluate_container
from pyai.layer.layer_learn import MemoryDecayType
from pyai.base.function_toolbox import normalize,spm_dot, nat_log,softmax
from pyai.base.miscellaneous_toolbox import isField
from pyai.base.miscellaneous_toolbox import isNone,flatten_last_n_dimensions,flexible_toString,flexible_print,flexible_copy


def controlability_task(modelname,savepath,rule = 'c1'):
    Nf = 1

    initial_state = 0
    D_ = normalize([np.array([1,1,1])])         #[Square, Triangle, Circle]
    d_ = [0.1*np.ones(D_[0].shape)]


    A_ = [np.eye(3)] # The observations are the current shape
    # No a_, the observations are perfect

    pb = 0.05
    
    # Line = where we're going
    # Column = where we're from
    B_ = [np.zeros((3,3,3))] # 3 from-states, 3 to-states, 3 possible actions (yellow, blue, purple)

    state_dependent_allowable_action = np.array([[1      ,0         ,1],     # Yellow
                                                [0      ,1         ,1],     # Blue
                                                [1      ,1         ,0]])   # Purple
                                                # Square | Triangle | Circle

    state_dependent_allowable_action = None

    if (rule=='c1'):
        # CONTROLABLE RULE 1
        # Action 1 : yellow -> triangle
        B_[0][:,:,0] = np.array(
                [[pb    ,pb    ,pb    ],
                 [1-2*pb,1-2*pb,1-2*pb],
                 [pb    ,pb    ,pb    ]]
        )
        # Action 2 : blue -> square
        B_[0][:,:,1] = np.array(
               [[1-2*pb,1-2*pb,1-2*pb],
                [pb    ,pb    ,pb    ],
                [pb    ,pb    ,pb    ]]
        )
        # Action 3 : purple -> circle
        B_[0][:,:,2] = np.array(
               [[pb    ,pb    ,pb    ],
                [pb    ,pb    ,pb    ],
                [1-2*pb,1-2*pb,1-2*pb]]
        )
    elif (rule=='c2'):
        # CONTROLABLE RULE 2
        # Action 1 : yellow -> square
        B_[0][:,:,0] = np.array(
                [[1-2*pb,1-2*pb,1-2*pb],
                    [pb    ,pb    ,pb    ],
                    [pb    ,pb    ,pb    ]]
        )
        # Action 2 : blue -> circle
        B_[0][:,:,1] = np.array(
                [[pb    ,pb    ,pb    ],
                    [pb    ,pb    ,pb    ],
                    [1-2*pb,1-2*pb,1-2*pb]]
        )
        # Action 3 : purple -> triangle
        B_[0][:,:,2] = np.array(
                [[pb    ,pb    ,pb    ],
                    [1-2*pb,1-2*pb,1-2*pb],
                    [pb    ,pb    ,pb    ]]
        )

        
    elif (rule=='u1'):
        # UNCONTROLABLE RULE 1
        # Action 1 : yellow -> square
        B_[0][:,:,0] = np.array(
                [[pb    ,pb    ,1-2*pb],
                 [1-2*pb,pb    ,pb    ],
                 [pb    ,1-2*pb,pb    ]]
        )
        B_[0][:,:,1] = np.array(
                [[pb    ,pb    ,1-2*pb],
                 [1-2*pb,pb    ,pb    ],
                 [pb    ,1-2*pb,pb    ]]
        )
        B_[0][:,:,2] = np.array(
                [[pb    ,pb    ,1-2*pb],
                 [1-2*pb,pb    ,pb    ],
                 [pb    ,1-2*pb,pb    ]]
        )

    elif (rule=='u2'):
        # UNCONTROLABLE RULE 2
        # Action 1 : yellow -> square
        B_[0][:,:,0] = np.array(
                [[pb    ,1-2*pb,pb    ],
                 [pb    ,pb    ,1-2*pb],
                 [1-2*pb,pb    ,pb    ]]
        )
        B_[0][:,:,1] = np.array(
                [[pb    ,1-2*pb,pb    ],
                 [pb    ,pb    ,1-2*pb],
                 [1-2*pb,pb    ,pb    ]]
        )
        B_[0][:,:,2] = np.array(
                [[pb    ,1-2*pb,pb    ],
                 [pb    ,pb    ,1-2*pb],
                 [1-2*pb,pb    ,pb    ]]
        )
    else : 
        print("Indicated rule " + rule + " hasn't been implemented")
        return

    b_ = [np.ones((B_[0].shape))]
    

    C_ = [np.array([[1],
                   [1],
                   [1]])]


    # Policies
    U_ = np.array([
        [0],
        [1],
        [2]
    ])

    #Habits
    E_ = None

    T = 10
    savemanager = active_model_save_manager.ActiveSaveManager(T,modelname=modelname,folder_name=savepath)
                                    # Trial related save , timestep related save
    nf_model = active_model.ActiveModel(savemanager,modelname,savepath)

    nf_model.T = T
    nf_model.A = A_
    nf_model.B = B_
    nf_model.b = b_
    nf_model.C = C_
    nf_model.D = D_
    nf_model.d = d_
    nf_model.U = U_
    nf_model.state_dependent_u = state_dependent_allowable_action

    nf_model.layer_options.learn_b = True
    nf_model.layer_options.learn_d = True

    nf_model.layer_options.T_horizon = 2
    nf_model.layer_options.learn_during_experience = False
    
    nf_model.layer_options.memory_decay = MemoryDecayType.NO_MEMORY_DECAY
    nf_model.layer_options.memory_decay = MemoryDecayType.STATIC
    nf_model.layer_options.decay_half_time = 500

    return nf_model


def predict_upcoming_obs_given_action(last_O,Prior_P,U,a,b,state_dependent_allowable_actions=None, verbose = False) :    
    # Nf = len(B) --> not because B is in Kronecker form
    Nf = a[0].ndim-1  # granted A has at least 1 modality, but any situation without observation isn't explored here
    Nmod = len(a)
    Prior_P = np.copy(Prior_P)

    # L is the posterior over hidden states based on last observations & likelihood (A & O)
    L = 1
    for modality in range (Nmod):
        L = L * spm_dot(a[modality],last_O[modality]) 
    
    # P is the posterior over hidden states at the current time t based on likelihoods and priors
    P_posterior =normalize(L.flatten()*Prior_P) # P(s|o) = P(o|s)P(s)  
        

    Q = []
    for action in range(U.shape[0]) :
        Q.append(np.dot(b[action],P_posterior)) # predictive posterior of states at time t depending on actions 
        
        for modality in range(Nmod):
            flattened_A = flatten_last_n_dimensions(a[modality].ndim-1,a[modality])
            qo = np.dot(flattened_A,Q[action]) # prediction over observations at time t
            print(qo)
    return Q


if __name__=="__main__":
    save_path = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","TEMPORARY_TEST_BED","CONTROLABILITY_TASK")
    overwrite = True
    Ntrials = 20
    model_name = "controlability_a"

    rule = 'c1'
    model = controlability_task(model_name,save_path,rule)
    model.initialize_n_layers(1)


    # possible_rules =["c1","u1","c2","u2"]
    # def pick_random_rule(possible_rules):
    #     return random.choice(possible_rules)

    # Nsec = 3
    # for number_of_sections in range(Nsec) :
    #     # rule = pick_random_rule(possible_rules)
    #     # standard_model = controlability_task("random","random",rule)
    #     # # model.B = flexible_copy(standard_model.B)
    #     # for lay in model.layer_list :
    #     #     lay.B_ = flexible_copy(standard_model.B)
    #     #     # print(lay.B_)
    #     model.add_n_trials(Ntrials,overwrite=overwrite)

    #     for ui in range(3):
    #         print('----------')
    #         print(model.layer_list[0].B_[0][:,:,ui])
    #         print(model.layer_list[0].b_[0][:,:,ui])
    
    # # complete_data = True
    # # var = True 
    # # save_model_performance_dictionnary(save_path,model_name,evaluate_container,overwrite=overwrite,include_var=var,include_complete=complete_data)
    # # full_dico = load_model_performance_dictionnary(save_path,model_name,var,complete_data)
    # from pyai.base.file_toolbox import load_flexible
    # from pyai.model.active_model_save_manager import ActiveSaveManager
    # from pyai.model.metrics import flexible_entropy,flexible_kl_dir
    # import statistics as stat

    # L = []
    # T = []
    # for inst in range(1):
    #     L.append([])
    #     T.append([])
    #     for trial in range(int(Nsec*Ntrials)):
    #         cont = ActiveSaveManager.open_trial_container(os.path.join(save_path,model_name),inst,trial)  
            
    #         K = (np.sum(np.abs(normalize(cont.b_[0])-(normalize(cont.B_[0])))))
    #         print(K)
    #         L[inst].append(K)
    #         T[inst].append(trial)

    # for i in range(Nsec):
    #     plt.axvline(Ntrials*i,color='r')
    # plt.plot(T[0],L[0])
    # plt.show()
    
