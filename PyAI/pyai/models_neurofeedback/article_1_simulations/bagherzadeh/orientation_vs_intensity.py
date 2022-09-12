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


def neurofeedback_model(modelname,savepath,p_i,p_o,
            neurofeedback_training_group = 'right',
            learn_a = True,prior_a_precision = 1.0,prior_a_confidence=1,
            learn_b=True,prior_b_precision = 1.0,prior_b_confidence=1,
            learn_d=True,
            mem_dec_type=MemoryDecayType.NO_MEMORY_DECAY,mem_dec_halftime=5000,
            perfect_a = False,perfect_b=False,perfect_d = False,
            verbose = False,SHAM="False"):
    Nf = 2 # two mental states are interesting in this scenario
    Ns = [5,3] # 5 orientation levels possible (left,center-left,middle,center-right,right) & 3 attention level possible (Low,Medium,High)
    D_ = [np.array([0,0,1,0,0]),np.array([0,1,0])] # Initial state (attention isn't focused either right or left, intensity is medium)
    D_ = normalize(D_)

    d_ =[]
    if (perfect_d):
        d_ = D_
        learn_d = False
    else :
        d_= [np.ones(D_[0].shape),np.ones(D_[1].shape)]
    # Neutral priors about the starting states

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # OBSERVATIONS : 
    # Generally : A[modality] is of shape (Number of outcomes for this modality) x (Number of states for 1st factor) x ... x (Number of states for nth factor)
    # Depending on the LNT or RNT, the A matrix is different :
    
    # Here, let's pick 5 feedback levels possible, equivalent to the
    # difference between the two states
    No = 5 # 1-very_bad 2-bad 3-neutral 4-good 5-very_good
    if SHAM=="False" :
        A_ = np.zeros((No,5,3))
        for attentive_level in range(Ns[1]):
            A_[:,:,attentive_level] = np.eye(5) # Whatever the attentional intensity, A_ is the same
            
            # attention is goood : 
            A_[:,:,0] = normalize(np.ones(A_[:,:,0].shape))
            A_[:,:,1] = normalize(np.ones(A_[:,:,0].shape) + 2*np.eye(5))
            A_[:,:,2] = np.eye(5)
        # Right is at zero :
        if (neurofeedback_training_group =="right"):
            # Left alpha is to be very distinct from right alpha
            # We model this with a simple state correlated to the level 
            # of alpha :
            # If right is low :

            nothing = "lol"

        elif (neurofeedback_training_group =="left"):
            # Left alpha is to be very distinct from right alpha
            # We model this with a simple state correlated to the level 
            # of alpha :
            A_ = np.rot90(A_,axes=(0,1))
            #A_ = 
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
    

    # FOR ORIENTATION
    to_the_right =np.array([[0,0,0,0,0],
                            [1,0,0,0,0],  
                            [0,1,0,0,0],  # Move towards the right from any state
                            [0,0,1,0,0], 
                            [0,0,0,1,1]]) 
    to_the_left = np.array([[1,1,0,0,0],
                            [0,0,1,0,0],  
                            [0,0,0,1,0],   # Move towards the left from any state
                            [0,0,0,0,1],
                            [0,0,0,0,0]])
    p_recenter = p_o
    orientation_decay = np.array([[1-p_recenter,0           ,0,0           ,0           ],
                                  [p_recenter  ,1-p_recenter,0,0           ,0           ],  
                                  [0           ,p_recenter  ,1,p_recenter  ,0           ],   
                                  [0           ,0           ,0,1-p_recenter,p_recenter  ],
                                  [0           ,0           ,0,0           ,1-p_recenter]])
    # If we focus on changing the attention level, there is a chance our attention direction will decay ?
    
    stayhere_activity = np.eye(Ns[0])
    random_orientation = normalize(np.ones(to_the_left.shape))


    # FOR INTENSITY
    concentrate = np.array([[0,0,0],
                            [1,0,0],
                            [0,1,1]])
    deconcentrate = np.array([[1,1,0],
                              [0,0,1],
                              [0,0,0]])
    p_deconcentrate = p_i
    natural_decay = np.array([[1,p_deconcentrate    ,0                ],
                              [0,1-p_deconcentrate  ,p_deconcentrate  ],
                              [0,0                  ,1-p_deconcentrate]])
    
    n_b = 7
    B_ = [np.zeros((Ns[0],Ns[0])+(n_b,)),np.zeros((Ns[1],Ns[1])+(3,))]
    B_[0][:,:,0] = to_the_right
    B_[0][:,:,1] = to_the_left
    B_[0][:,:,2] = stayhere_activity
    B_[0][:,:,3] = orientation_decay
    for k in range(4,n_b):
        B_[0][:,:,k] = random_orientation
    
    B_[1][:,:,0] = natural_decay
    B_[1][:,:,1] = concentrate
    B_[1][:,:,2] = deconcentrate

    if (perfect_b):
        b_ = B_
        learn_b = False
    else :
        b_ = [0,0]
        b_[0] = prior_b_confidence*(np.ones(B_[0].shape) + (1-prior_b_precision)*B_[0])
        b_[1] = prior_b_confidence*(np.ones(B_[1].shape) + (1-prior_b_precision)*B_[1])

    # U_ = np.zeros((nu,len(Ns)))
    U_ = np.array([[0,0],    # To the right + nat decay
                   [1,0],    # To the left + nat decay
                   [2,0],    # Remain here + nat decay
                   [3,1],    # orient decay + concentrate
                   [3,2],    # orient decay + deconcentrate
                   [4,0],    # Random action + nat decay
                   [5,0],    # Random action + nat decay
                   [6,0]])   # Random action + nat decay
    
    
    la = -2
    rs = 2
    C_mental = np.array([[2*la],
                        [1*la],
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


def neurofeedback_model_2(modelname,savepath,p_i,p_o,
            neurofeedback_training_group = 'right',
            learn_a = True,prior_a_precision = 1.0,prior_a_confidence=1,
            learn_b=True,prior_b_precision = 1.0,prior_b_confidence=1,
            learn_d=True,prior_d_precision = 1.0,prior_d_confidence=1,
            mem_dec_type=MemoryDecayType.NO_MEMORY_DECAY,mem_dec_halftime=5000,
            perfect_a = False,perfect_b=False,perfect_d = False,
            verbose = False,SHAM="False"):
    Nf = 2 # two mental states are interesting in this scenario
    Ns = [5,3] # 5 orientation levels possible (left,center-left,middle,center-right,right) & 3 attention level possible (Low,Medium,High)
    D_ = [np.array([0,0,1,0,0]),np.array([0,0,1])] # Initial state (attention isn't focused either right or left, intensity is medium)
    D_ = normalize(D_)

    d_ =[]
    if (perfect_d):
        d_ = D_
        learn_d = False
    else :
        d_ = [None,None]
        d_[0] = prior_d_confidence*(np.ones(D_[0].shape) + (1-prior_d_precision)*D_[0])
        d_[1] = prior_d_confidence*(np.ones(D_[1].shape) + (1-prior_d_precision)*D_[1])
    # Neutral priors about the starting states

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # OBSERVATIONS : 
    # Generally : A[modality] is of shape (Number of outcomes for this modality) x (Number of states for 1st factor) x ... x (Number of states for nth factor)
    # Depending on the LNT or RNT, the A matrix is different :
    
    # Here, let's pick 5 feedback levels possible, equivalent to the
    # difference between the two states
    No = 5 # 1-very_bad 2-bad 3-neutral 4-good 5-very_good
    if SHAM=="False" :
        A_ = np.zeros((No,5,3))
        for attentive_level in range(Ns[1]):
            A_[:,:,attentive_level] = np.eye(5) # Whatever the attentional intensity, A_ is the same
            
            # attention is goood : 
            A_[:,:,0] = (np.ones(A_[:,:,0].shape))
            A_[:,:,1] = (np.ones(A_[:,:,0].shape) + 2*np.eye(5))
            A_[:,:,2] = np.eye(5)
        # Right is at zero :
        if (neurofeedback_training_group =="right"):
            # Left alpha is to be very distinct from right alpha
            # We model this with a simple state correlated to the level 
            # of alpha :
            # If right is low :

            nothing = "lol"

        elif (neurofeedback_training_group =="left"):
            # Left alpha is to be very distinct from right alpha
            # We model this with a simple state correlated to the level 
            # of alpha :
            A_ = np.rot90(A_,axes=(0,1))
            #A_ = 
        A_ = normalize([A_])
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
    

    # FOR ORIENTATION
    to_the_right =np.array([[0,0,0,0,0],
                            [1,0,0,0,0],  
                            [0,1,0,0,0],  # Move towards the right from any state
                            [0,0,1,0,0], 
                            [0,0,0,1,1]]) 
    to_the_left = np.array([[1,1,0,0,0],
                            [0,0,1,0,0],  
                            [0,0,0,1,0],   # Move towards the left from any state
                            [0,0,0,0,1],
                            [0,0,0,0,0]])
    p_recenter = p_o
    orientation_decay = np.array([[1-p_recenter,0           ,0,0           ,0           ],
                                  [p_recenter  ,1-p_recenter,0,0           ,0           ],  
                                  [0           ,p_recenter  ,1,p_recenter  ,0           ],   
                                  [0           ,0           ,0,1-p_recenter,p_recenter  ],
                                  [0           ,0           ,0,0           ,1-p_recenter]])
    # If we focus on changing the attention level, there is a chance our attention direction will decay ?
    
    stayhere_activity = np.eye(Ns[0])
    random_orientation = normalize(np.ones(to_the_left.shape))


    # FOR INTENSITY
    concentrate = np.array([[0,0,0],
                            [1,0,0],
                            [0,1,1]])
    deconcentrate = np.array([[1,1,0],
                              [0,0,1],
                              [0,0,0]])
    p_deconcentrate = p_i
    natural_decay = np.array([[1,p_deconcentrate    ,0                ],
                              [0,1-p_deconcentrate  ,p_deconcentrate  ],
                              [0,0                  ,1-p_deconcentrate]])
    
    n_b = 7
    B_ = [np.zeros((Ns[0],Ns[0])+(n_b,)),np.zeros((Ns[1],Ns[1])+(3,))]
    B_[0][:,:,0] = to_the_right
    B_[0][:,:,1] = to_the_left
    B_[0][:,:,2] = stayhere_activity
    B_[0][:,:,3] = orientation_decay
    for k in range(4,n_b):
        B_[0][:,:,k] = random_orientation
    
    B_[1][:,:,0] = natural_decay
    B_[1][:,:,1] = concentrate
    B_[1][:,:,2] = deconcentrate

    if (perfect_b):
        b_ = B_
        learn_b = False
    else :
        b_ = [0,0]
        b_[0] = prior_b_confidence*(np.ones(B_[0].shape) + (1-prior_b_precision)*B_[0])
        b_[1] = prior_b_confidence*(np.ones(B_[1].shape) + (1-prior_b_precision)*B_[1])

    # U_ = np.zeros((nu,len(Ns)))
    U_ = np.array([[0,0],    # To the right + nat decay
                   [1,0],    # To the left + nat decay
                   [2,0],    # Remain here + nat decay
                   [3,1],    # orient decay + concentrate
                   [3,2],    # orient decay + deconcentrate
                   [4,0],    # Random action + nat decay
                   [5,0],    # Random action + nat decay
                   [6,0]])   # Random action + nat decay
    
    
    la = -2
    rs = 2
    C_mental = np.array([[2*la],
                        [1*la],
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


def neurofeedback_model_3(modelname,savepath,p_i,p_o,
            neurofeedback_training_group = 'right',
            learn_a = True,prior_a_precision = 1.0,prior_a_confidence=1,
            learn_b=True,prior_b_precision = 1.0,prior_b_confidence=1,
            learn_d=True,prior_d_precision = 1.0,prior_d_confidence=1,
            mem_dec_type=MemoryDecayType.NO_MEMORY_DECAY,mem_dec_halftime=5000,
            perfect_a = False,perfect_b=False,perfect_d = False,
            verbose = False,SHAM="False"):
    Nf = 2 # two mental states are interesting in this scenario
    Ns = [5,3] # 5 orientation levels possible (left,center-left,middle,center-right,right) & 3 attention level possible (Low,Medium,High)
    D_ = [np.array([0,0,1,0,0]),np.array([0,0,1])] # Initial state (attention isn't focused either right or left, intensity is medium)
    D_ = normalize(D_)

    d_ =[]
    if (perfect_d):
        d_ = D_
        learn_d = False
    else :
        d_ = [None,None]
        d_[0] = prior_d_confidence*(np.ones(D_[0].shape) + (prior_d_precision-1)*D_[0])
        d_[1] = prior_d_confidence*(np.ones(D_[1].shape) + (prior_d_precision-1)*D_[1])
    # Neutral priors about the starting states

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # OBSERVATIONS : 
    # Generally : A[modality] is of shape (Number of outcomes for this modality) x (Number of states for 1st factor) x ... x (Number of states for nth factor)
    # Depending on the LNT or RNT, the A matrix is different :
    
    # Here, let's pick 5 feedback levels possible, equivalent to the
    # difference between the two states
    No = 5 # 1-very_bad 2-bad 3-neutral 4-good 5-very_good
    if SHAM=="False" :
        A_ = np.zeros((No,5,3))
        for attentive_level in range(Ns[1]):
            A_[:,:,attentive_level] = np.eye(5) # Whatever the attentional intensity, A_ is the same
            
            # attention is goood : 
            A_[:,:,0] = (np.ones(A_[:,:,0].shape))
            A_[:,:,1] = (np.ones(A_[:,:,0].shape) + 2*np.eye(5))
            A_[:,:,2] = np.eye(5)



            A_[:,:,0] = (np.ones(A_[:,:,0].shape))
            A_[:,:,1] = (np.ones(A_[:,:,0].shape) + 2*np.eye(5))
            A_[:,:,2] = np.eye(5)


        # Right is at zero :
        if (neurofeedback_training_group =="right"):
            # Left alpha is to be very distinct from right alpha
            # We model this with a simple state correlated to the level 
            # of alpha :
            # If right is low :

            nothing = "lol"
            A_[:,:,0] = np.array([[1,1,1,1,1],
                                  [0,0,0,0,0],
                                  [0,0,0,0,0],
                                  [0,0,0,0,0],
                                  [0,0,0,0,0]])
            A_[:,:,1] = np.array([[1,1,1,1,1],
                                  [0,0,0,0,0],
                                  [0,0,0,0,0],
                                  [0,0,0,0,0],
                                  [0,0,0,0,0]])
            A_[:,:,1] = A_[:,:,1] + np.eye(5)
            A_[:,:,2] = np.eye(5)

        elif (neurofeedback_training_group =="left"):
            # Left alpha is to be very distinct from right alpha
            # We model this with a simple state correlated to the level 
            # of alpha :
            A_[:,:,0] = np.array([[1,1,1,1,1],
                                  [0,0,0,0,0],
                                  [0,0,0,0,0],
                                  [0,0,0,0,0],
                                  [0,0,0,0,0]])
            A_[:,:,1] = np.array([[1,1,1,1,2],
                                  [0,0,0,1,0],
                                  [0,0,1,0,0],
                                  [0,1,0,0,0],
                                  [1,0,0,0,0]])
            A_[:,:,2] = np.array([[0,0,0,0,1],
                                  [0,0,0,1,0],
                                  [0,0,1,0,0],
                                  [0,1,0,0,0],
                                  [1,0,0,0,0]])
            #A_ = 
            #A_ = np.rot90(A_,axes=(0,1))
        A_ = normalize([A_])
    else :
        A_ = [normalize((No,)+tuple(Ns))]  
    
    # prior_a_sigma : true values are the mean of our model
    # prior_strength : Base weight --> The higher this number, the stronger priors are and the longer it takes for experience to "drown" them \in [0,+OO[
    
    if (not(perfect_a)):
        # a priors are flat to begin with ?
        a_ = [0]
        a_[0] = prior_a_confidence*(np.ones(A_[0].shape) + (prior_a_precision-1)*A_[0])
    else : 
        a_ = A_
        learn_a=False
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # ACTIONS :
    # Transition matrixes between hidden states ( = control states)
    # B_ = climb_stairs_B(pb=1,npoub=npoubelle)
    
    to_the_right_sharp =np.array([[0,0,0,0,0],
                                  [0,0,0,0,0],  
                                  [1,0,0,0,0],  # Move towards the right from any state
                                  [0,1,0,0,0], 
                                  [0,0,1,1,1]]) 
    to_the_left_sharp = np.array([[1,1,1,0,0],
                                  [0,0,0,1,0],  
                                  [0,0,0,0,1],   # Move towards the left from any state
                                  [0,0,0,0,0],
                                  [0,0,0,0,0]])
    # FOR ORIENTATION
    to_the_right =np.array([[0,0,0,0,0],
                            [1,0,0,0,0],  
                            [0,1,0,0,0],  # Move towards the right from any state
                            [0,0,1,0,0], 
                            [0,0,0,1,1]]) 
    to_the_left = np.array([[1,1,0,0,0],
                            [0,0,1,0,0],  
                            [0,0,0,1,0],   # Move towards the left from any state
                            [0,0,0,0,1],
                            [0,0,0,0,0]])
    
    p_recenter = p_o
    orientation_decay = np.array([[1-p_recenter,0           ,0,0           ,0           ],
                                  [p_recenter  ,1-p_recenter,0,0           ,0           ],  
                                  [0           ,p_recenter  ,1,p_recenter  ,0           ],   
                                  [0           ,0           ,0,1-p_recenter,p_recenter  ],
                                  [0           ,0           ,0,0           ,1-p_recenter]])
    # If we focus on changing the attention level, there is a chance our attention direction will decay ?
    
    stayhere_activity = np.eye(Ns[0])
    random_orientation = normalize(np.ones(to_the_left.shape))


    # FOR INTENSITY
    concentrate = np.array([[0,0,0],
                            [1,0,0],
                            [0,1,1]])
    deconcentrate = np.array([[1,1,0],
                              [0,0,1],
                              [0,0,0]])
    p_deconcentrate = p_i
    natural_decay = np.array([[1,p_deconcentrate    ,0                ],
                              [0,1-p_deconcentrate  ,p_deconcentrate  ],
                              [0,0                  ,1-p_deconcentrate]])
    p_deconcentrate_heavy = 2*p_i
    heavy_decay = np.array([[1,p_deconcentrate    ,0                ],
                              [0,1-p_deconcentrate  ,p_deconcentrate  ],
                              [0,0                  ,1-p_deconcentrate]])
    
    n_b = 9
    B_ = [np.zeros((Ns[0],Ns[0])+(n_b,)),np.zeros((Ns[1],Ns[1])+(3,))]
    B_[0][:,:,0] = to_the_right
    B_[0][:,:,1] = to_the_left
    B_[0][:,:,2] = stayhere_activity
    B_[0][:,:,3] = orientation_decay
    B_[0][:,:,4] = to_the_right_sharp
    B_[0][:,:,5] = to_the_left_sharp
    for k in range(6,n_b):
        B_[0][:,:,k] = random_orientation
    
    B_[1][:,:,0] = natural_decay
    B_[1][:,:,1] = concentrate
    B_[1][:,:,2] = deconcentrate

    if (perfect_b):
        b_ = B_
        learn_b = False
    else :
        b_ = [0,0]
        b_[0] = prior_b_confidence*(np.ones(B_[0].shape) + (prior_b_precision-1)*B_[0])
        b_[1] = prior_b_confidence*(np.ones(B_[1].shape) + (prior_b_precision-1)*B_[1])

    # U_ = np.zeros((nu,len(Ns)))
    U_ = np.array([[0,0],    # To the right + nat decay
                   [1,0],    # To the left + nat decay
                   [4,0],    # To the right sharp + nat decay
                   [5,0],    # To the left sharp + nat decay
                   [2,0],    # Remain here + nat decay
                   [3,1],    # orient decay + concentrate
                   [3,2],    # orient decay + deconcentrate
                   [4,0],    # Random action + nat decay
                   [5,0],    # Random action + nat decay
                   [6,0]])   # Random action + nat decay
    
    
    la = -2
    rs = 2
    C_mental = np.array([[2*la],
                        [1*la],
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


def neurofeedback_model_intensityuseless(modelname,savepath,p_i,p_o,
            neurofeedback_training_group = 'right',
            learn_a = True,prior_a_precision = 1.0,prior_a_confidence=1,
            learn_b=True,prior_b_precision = 1.0,prior_b_confidence=1,
            learn_d=True,prior_d_precision = 1.0,prior_d_confidence=1,
            mem_dec_type=MemoryDecayType.NO_MEMORY_DECAY,mem_dec_halftime=5000,
            perfect_a = False,perfect_b=False,perfect_d = False,
            verbose = False,SHAM="False"):
    Nf = 2 # two mental states are interesting in this scenario
    Ns = [5,3] # 5 orientation levels possible (left,center-left,middle,center-right,right) & 3 attention level possible (Low,Medium,High)
    D_ = [np.array([0,0,1,0,0]),np.array([1,1,1])] # Initial state (attention isn't focused either right or left, intensity is medium)
    D_ = normalize(D_)

    d_ =[]
    if (perfect_d):
        d_ = D_
        learn_d = False
    else :
        d_ = [None,None]
        d_[0] = prior_d_confidence*(np.ones(D_[0].shape) + (prior_d_precision-1)*D_[0])
        d_[1] = prior_d_confidence*(np.ones(D_[1].shape) + (prior_d_precision-1)*D_[1])
    # Neutral priors about the starting states

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # OBSERVATIONS : 
    # Generally : A[modality] is of shape (Number of outcomes for this modality) x (Number of states for 1st factor) x ... x (Number of states for nth factor)
    # Depending on the LNT or RNT, the A matrix is different :
    
    # Here, let's pick 5 feedback levels possible, equivalent to the
    # difference between the two states
    No = 5 # 1-very_bad 2-bad 3-neutral 4-good 5-very_good
    if SHAM=="False" :
        A_ = np.zeros((No,5,3))
        for attentive_level in range(Ns[1]):
            A_[:,:,attentive_level] = np.eye(5) # Whatever the attentional intensity, A_ is the same
            
            # attention is goood : 
            A_[:,:,0] = (np.ones(A_[:,:,0].shape))
            A_[:,:,1] = (np.ones(A_[:,:,0].shape) + 2*np.eye(5))
            A_[:,:,2] = np.eye(5)



            A_[:,:,0] = (np.ones(A_[:,:,0].shape))
            A_[:,:,1] = (np.ones(A_[:,:,0].shape) + 2*np.eye(5))
            A_[:,:,2] = np.eye(5)


        # Right is at zero :
        if (neurofeedback_training_group =="right"):
            for i in range(3):
                A_[:,:,i] = np.eye(5)

        elif (neurofeedback_training_group =="left"):
            for i in range(3):
                A_[:,:,i] = np.array([[0,0,0,0,1],
                                    [0,0,0,1,0],
                                    [0,0,1,0,0],
                                    [0,1,0,0,0],
                                    [1,0,0,0,0]])
            #A_ = 
            #A_ = np.rot90(A_,axes=(0,1))
        A_ = normalize([A_])
    else :
        A_ = [normalize(np.ones((No,)+tuple(Ns)))]  
    
    # prior_a_sigma : true values are the mean of our model
    # prior_strength : Base weight --> The higher this number, the stronger priors are and the longer it takes for experience to "drown" them \in [0,+OO[
    
    if (not(perfect_a)):
        # a priors are flat to begin with ?
        a_ = [0]
        a_[0] = prior_a_confidence*(np.ones(A_[0].shape) + (prior_a_precision-1)*A_[0])
    else : 
        a_ = A_
        learn_a=False
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # ACTIONS :
    # Transition matrixes between hidden states ( = control states)
    # B_ = climb_stairs_B(pb=1,npoub=npoubelle)
    
    to_the_right_sharp =np.array([[0,0,0,0,0],
                                  [0,0,0,0,0],  
                                  [1,0,0,0,0],  # Move towards the right from any state
                                  [0,1,0,0,0], 
                                  [0,0,1,1,1]]) 
    to_the_left_sharp = np.array([[1,1,1,0,0],
                                  [0,0,0,1,0],  
                                  [0,0,0,0,1],   # Move towards the left from any state
                                  [0,0,0,0,0],
                                  [0,0,0,0,0]])
    # FOR ORIENTATION
    to_the_right =np.array([[0,0,0,0,0],
                            [1,0,0,0,0],  
                            [0,1,0,0,0],  # Move towards the right from any state
                            [0,0,1,0,0], 
                            [0,0,0,1,1]]) 
    to_the_left = np.array([[1,1,0,0,0],
                            [0,0,1,0,0],  
                            [0,0,0,1,0],   # Move towards the left from any state
                            [0,0,0,0,1],
                            [0,0,0,0,0]])
    
    p_recenter = p_o
    orientation_decay = np.array([[1-p_recenter,0           ,0,0           ,0           ],
                                  [p_recenter  ,1-p_recenter,0,0           ,0           ],  
                                  [0           ,p_recenter  ,1,p_recenter  ,0           ],   
                                  [0           ,0           ,0,1-p_recenter,p_recenter  ],
                                  [0           ,0           ,0,0           ,1-p_recenter]])
    # If we focus on changing the attention level, there is a chance our attention direction will decay ?
    
    stayhere_activity = np.eye(Ns[0])
    random_orientation = normalize(np.ones(to_the_left.shape))


    # FOR INTENSITY
    concentrate = np.array([[0,0,0],
                            [1,0,0],
                            [0,1,1]])
    deconcentrate = np.array([[1,1,0],
                              [0,0,1],
                              [0,0,0]])
    p_deconcentrate = p_i
    natural_decay = np.array([[1,p_deconcentrate    ,0                ],
                              [0,1-p_deconcentrate  ,p_deconcentrate  ],
                              [0,0                  ,1-p_deconcentrate]])
    
    n_b = 9
    B_ = [np.zeros((Ns[0],Ns[0])+(n_b,)),np.zeros((Ns[1],Ns[1])+(3,))]
    B_[0][:,:,0] = to_the_right
    B_[0][:,:,1] = to_the_left
    B_[0][:,:,2] = stayhere_activity
    B_[0][:,:,3] = orientation_decay
    B_[0][:,:,4] = to_the_right_sharp
    B_[0][:,:,5] = to_the_left_sharp
    for k in range(6,n_b):
        B_[0][:,:,k] = random_orientation
    
    B_[1][:,:,0] = natural_decay
    B_[1][:,:,1] = concentrate
    B_[1][:,:,2] = deconcentrate

    if (perfect_b):
        b_ = B_
        learn_b = False
    else :
        b_ = [0,0]
        b_[0] = prior_b_confidence*(np.ones(B_[0].shape) + (prior_b_precision-1)*B_[0])
        b_[1] = prior_b_confidence*(np.ones(B_[1].shape) + (prior_b_precision-1)*B_[1])

    # U_ = np.zeros((nu,len(Ns)))
    U_ = np.array([[0,0],    # To the right + nat decay
                   [1,0],    # To the left + nat decay
                   [4,0],    # To the right sharp + nat decay
                   [5,0],    # To the left sharp + nat decay
                   [2,0],    # Remain here + nat decay
                   [3,1],    # orient decay + concentrate
                   [3,2],    # orient decay + deconcentrate
                   [4,0],    # Random action + nat decay
                   [5,0],    # Random action + nat decay
                   [6,0]])   # Random action + nat decay
    
    
    la = -2
    rs = 2
    C_mental = np.array([[2*la],
                        [1*la],
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


def bagherzadeh_model(modelname,savepath,p_i,p_o,
            neurofeedback_training_group = 'right',
            learn_a = True,prior_a_precision = 1.0,prior_a_confidence=1,
            learn_b=True,prior_b_precision = 1.0,prior_b_confidence=1,
            learn_d=True,prior_d_precision = 1.0,prior_d_confidence=1,
            mem_dec_type=MemoryDecayType.NO_MEMORY_DECAY,mem_dec_halftime=5000,
            perfect_a = False,perfect_b=False,perfect_d = False,
            verbose = False,SHAM="False"):
    Nf = 2 # two mental states are interesting in this scenario
    Ns = [5,3] # 5 orientation levels possible (left,center-left,middle,center-right,right) & 3 attention level possible (Low,Medium,High)
    D_ = [np.array([0,0,1,0,0]),np.array([1,1,1])] # Initial state (attention isn't focused either right or left, intensity is medium)
    D_ = normalize(D_)

    d_ =[]
    if (perfect_d):
        d_ = D_
        learn_d = False
    else :
        d_ = [None,None]
        d_[0] = prior_d_confidence*(np.ones(D_[0].shape) + (prior_d_precision-1)*D_[0])
        d_[1] = prior_d_confidence*(np.ones(D_[1].shape) + (prior_d_precision-1)*D_[1])
    # Neutral priors about the starting states

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # OBSERVATIONS : 
    # Generally : A[modality] is of shape (Number of outcomes for this modality) x (Number of states for 1st factor) x ... x (Number of states for nth factor)
    # Depending on the LNT or RNT, the A matrix is different :
    
    # Here, let's pick 5 feedback levels possible, equivalent to the
    # difference between the two states
    No = 5 # 1-very_bad 2-bad 3-neutral 4-good 5-very_good
    if SHAM=="False" :
        A_ = np.zeros((No,5,3))
        # for attentive_level in range(Ns[1]):
        #     A_[:,:,attentive_level] = np.eye(5) # Whatever the attentional intensity, A_ is the same
            
        #     # attention is goood : 
        #     A_[:,:,0] = (np.ones(A_[:,:,0].shape))
        #     A_[:,:,1] = (np.ones(A_[:,:,0].shape) + 2*np.eye(5))
        #     A_[:,:,2] = np.eye(5)



        #     A_[:,:,0] = (np.ones(A_[:,:,0].shape))
        #     A_[:,:,1] = (np.ones(A_[:,:,0].shape) + 2*np.eye(5))
        #     A_[:,:,2] = np.eye(5)


        # # Right is at zero :
        # if (neurofeedback_training_group =="right"):
        #     A_[:,:,2] = np.array([[1,0,0,0,0],
        #                           [0,1,0,0,0],
        #                           [0,0,1,0,0],
        #                           [0,0,0,1,0],
        #                           [0,0,0,0,1]])
        #     A_[:,:,0] = normalize(np.ones((5,5)))
        #     A_[:,:,1] = normalize(np.ones((5,5))+4*A_[:,:,2])
        #     A_[:,:,2] = normalize(np.ones((5,5))+10*A_[:,:,2])

        # elif (neurofeedback_training_group =="left"):
        #     A_[:,:,2] = np.array([[0,0,0,0,1],
        #                           [0,0,0,1,0],
        #                           [0,0,1,0,0],
        #                           [0,1,0,0,0],
        #                           [1,0,0,0,0]])
        #     A_[:,:,0] = normalize(np.ones((5,5)))
        #     A_[:,:,1] = normalize(np.ones((5,5))+4*A_[:,:,2])
        #     A_[:,:,2] = normalize(np.ones((5,5))+10*A_[:,:,2])
        #     #A_ = 
        #     #A_ = np.rot90(A_,axes=(0,1))

        # Right is at zero :
        if (neurofeedback_training_group =="right"):
            for i in range(3):
                A_[:,:,i] = np.eye(5)

        elif (neurofeedback_training_group =="left"):
            for i in range(3):
                A_[:,:,i] = np.array([[0,0,0,0,1],
                                    [0,0,0,1,0],
                                    [0,0,1,0,0],
                                    [0,1,0,0,0],
                                    [1,0,0,0,0]])
        A_ = normalize([A_])
    else :
        A_ = [normalize(np.ones((No,)+tuple(Ns)))]  
    
    # prior_a_sigma : true values are the mean of our model
    # prior_strength : Base weight --> The higher this number, the stronger priors are and the longer it takes for experience to "drown" them \in [0,+OO[
    
    if (not(perfect_a)):
        # a priors are flat to begin with ?
        a_ = [0]
        a_[0] = prior_a_confidence*(np.ones(A_[0].shape) + (prior_a_precision-1)*A_[0])
    else : 
        a_ = A_
        learn_a=False
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # ACTIONS :
    # Transition matrixes between hidden states ( = control states)
    # B_ = climb_stairs_B(pb=1,npoub=npoubelle)

    # FOR ORIENTATION
    p_recenter = p_o
    to_the_right =np.array([[0,0,0           ,0           ,0           ],
                            [1,0,0           ,0           ,0           ],  
                            [0,1,p_recenter  ,0           ,0           ],  # Move towards the right from any state
                            [0,0,1-p_recenter,p_recenter  ,p_recenter  ], 
                            [0,0,0           ,1-p_recenter,1-p_recenter]]) 
    to_the_left = np.array([[1-p_recenter,1-p_recenter,0           ,0,0],
                            [p_recenter  ,p_recenter  ,1-p_recenter,0,0],  
                            [0           ,0           ,p_recenter  ,1,0],   # Move towards the left from any state
                            [0           ,0           ,0           ,0,1],
                            [0           ,0           ,0           ,0,0]])
    
    stayhere_activity = np.array([[1-p_recenter,0           ,0          ,0           ,0           ],
                                  [p_recenter  ,1-p_recenter,0          ,0           ,0           ],  
                                  [0           ,p_recenter  ,1          ,p_recenter  ,0           ],   # Move towards the left from any state
                                  [0           ,0           ,0          ,1-p_recenter,p_recenter  ],
                                  [0           ,0           ,0          ,0           ,1-p_recenter]])
    # If we focus on changing the attention level, there is a chance our attention direction will decay ?
    
    random_orientation = normalize(np.ones(to_the_left.shape))


    # FOR INTENSITY
    p_deconcentrate = p_i
    concentrate = np.array([[p_deconcentrate  ,0                ,0                ],
                            [1-p_deconcentrate,p_deconcentrate  ,p_deconcentrate  ],
                            [0                ,1-p_deconcentrate,1-p_deconcentrate]])
    deconcentrate = np.array([[1,1,0                ],
                              [0,0,p_deconcentrate  ],
                              [0,0,1-p_deconcentrate]])
    natural_decay = np.array([[1,p_deconcentrate    ,0                ],
                              [0,1-p_deconcentrate  ,p_deconcentrate  ],
                              [0,0                  ,1-p_deconcentrate]])
    
    n_b = 3
    B_ = [np.zeros((Ns[0],Ns[0])+(n_b,)),np.zeros((Ns[1],Ns[1])+(3,))]
    B_[0][:,:,0] = to_the_right
    B_[0][:,:,1] = to_the_left
    B_[0][:,:,2] = stayhere_activity
    
    B_[1][:,:,0] = natural_decay
    B_[1][:,:,1] = concentrate
    B_[1][:,:,2] = deconcentrate

    if (perfect_b):
        b_ = B_
        learn_b = False
    else :
        b_ = [0,0]
        b_[0] = prior_b_confidence*(np.ones(B_[0].shape) + (prior_b_precision-1)*B_[0])
        b_[1] = prior_b_confidence*(np.ones(B_[1].shape) + (prior_b_precision-1)*B_[1])
    # U_ = np.zeros((nu,len(Ns)))
    U_ = np.array([[0,0],    # To the right
                   [1,0],    # To the left            + nat decay
                   [2,0],    # stayhere_activity
                   [0,1],    # To the right
                   [1,1],    # To the left            + concentrate
                   [2,1],    # stayhere_activity
                   [0,2],    # To the right
                   [1,2],    # To the left            + deconcentrate
                   [2,2]])   # stayhere_activity
    
    
    la = -2
    rs = 2
    C_mental = np.array([[2*la],
                        [1*la],
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