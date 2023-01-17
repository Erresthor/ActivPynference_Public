import sys,os

from cmath import pi
from turtle import width
from imageio import save
import numpy as np
import matplotlib.pyplot as plt

from pyai.base.function_toolbox import *

from pyai.model import active_model,active_model_container,active_model_save_manager
from pyai.neurofeedback_run import save_model_performance_dictionnary,load_model_performance_dictionnary
from pyai.models_neurofeedback.article_1_simulations.climb_stairs_flat_priors import nf_model,nf_model_imp2,nf_model_imp,evaluate_container
from pyai.layer.layer_learn import MemoryDecayType

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

        state_dependent_allowable_action = np.array([[1      ,0         ,1],     # Yellow
                                                     [0      ,1         ,1],     # Blue
                                                     [1      ,1         ,0]])   # Purple
                                                    # Square | Triangle | Circle
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

        state_dependent_allowable_action = np.array([[0      ,1         ,1],     # Yellow
                                                     [1      ,1         ,0],     # Blue
                                                     [1      ,0         ,1]])   # Purple
                                                    # Square | Triangle | Circle
    elif (rule=='u1'):
        # UNCONTROLABLE RULE 1
        # Action 1 : yellow -> square
        B_[0][:,:,:] = np.array(
                [[pb    ,pb    ,1-2*pb],
                 [1-2*pb,pb    ,pb    ],
                 [pb    ,1-2*pb,pb    ]]
        )
        state_dependent_allowable_action = np.array([[0      ,1         ,1],     # Yellow
                                                     [1      ,1         ,0],     # Blue
                                                     [1      ,0         ,1]])   # Purple
                                                    # Square | Triangle | Circle
    elif (rule=='u2'):
        # UNCONTROLABLE RULE 2
        # Action 1 : yellow -> square
        B_[0][:,:,:] = np.array(
                [[pb    ,1-2*pb,pb    ],
                 [pb    ,pb    ,1-2*pb],
                 [1-2*pb,pb    ,pb    ]]
        )
    else : 
        print("Indicated rule " + rule + " hasn't been implemented")

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
    #nf_model.layer_options.memory_decay = MemoryDecayType.STATIC
    nf_model.layer_options.decay_half_time = 2000

    return nf_model


if __name__=="__main__":
    save_path = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","TEMPORARY_TEST_BED","CONTROLABILITY_TASK")
    overwrite = False
    Ntrials = 10
    model_name = "controlability_a"

    rule = 'c1'
    model = controlability_task(model_name,save_path,rule)
    model.initialize_n_layers(10)
    
    for rule in ["c1","u1","c2","u2"] :
        standard_model = controlability_task("random","random",rule)
        for lay in model.layer_list :
            lay.B_ = np.copy(standard_model.B)
        model.run_n_trials(Ntrials)
        print(model.layer_list[0].b_[0][:,:,2])
    
    # complete_data = True
    # var = True 
    # save_model_performance_dictionnary(save_path,model_name,evaluate_container,overwrite=overwrite,include_var=var,include_complete=complete_data)
    # full_dico = load_model_performance_dictionnary(save_path,model_name,var,complete_data)

    