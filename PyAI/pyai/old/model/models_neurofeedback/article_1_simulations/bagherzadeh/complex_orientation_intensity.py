from turtle import right
import numpy as np
import statistics as stat

from ....model.metrics import flexible_entropy,flexible_kl_dir

from ....layer_old.layer_learn import MemoryDecayType
from ....base.miscellaneous_toolbox import isField
from ....base.function_toolbox import normalize
from ....base.matrix_functions import matrix_distance_list,argmean

from ....model.active_model import ActiveModel
from ....model.active_model_save_manager import ActiveSaveManager
from ....base.normal_distribution_matrix import generate_normal_dist_along_matrix


def alpha_assymetry_model(modelname,savepath,
            neurofeedback_training_group = 'R',
            mem_dec_type=MemoryDecayType.NO_MEMORY_DECAY,mem_dec_halftime=5000,
            SHAM="False"):
    Pos = [0.25,0.5,1.0] # Probability of orientation decay dpending on int
    Pis = [0.3,0.3,0.3,0.3,0.3] # Probability of intensity decay dpending on int

    Nori = 5
    Nint = 3
    Ns = Nori*Nint
    Nfeedback = 5
    Nu = 3

    A = np.zeros((Nfeedback,Ns))
    B = np.zeros((Ns,Ns,Nu))
    C = np.zeros((Nfeedback,))
    D = np.zeros((Ns,))
    E = np.zeros((Nu,))

    state_counter = 0 # State counter = Iintens*Nint + Iori

    def state_index(intensity_index,orientation_index):
        if (intensity_index < 0) or (intensity_index>=Nint) or (orientation_index <0) or (orientation_index>= Nori):
            return "This will cause an error haha"
        return intensity_index*Nint + orientation_index

    for intensity_index in range(Nint):
        for orientation_index in range(Nori):
            current_state = state_index(intensity_index,orientation_index)
            p_o = Pos[intensity_index]
            p_i = Pis[orientation_index]


            ori_middle = int((Nori-1)/2)
            if orientation_index == ori_middle:
                D[state_counter] = 1 # Attention orientation always starts at the center
            
            # Where to move from the current state : 
            # First, natural attentional decay :
            #ORIENTATION ONLY
            left_state = state_index(intensity_index,orientation_index-1)
            right_state = state_index(intensity_index,orientation_index+1)
            if (orientation_index<ori_middle):
                # I'm on the left, decay leads me towards the right
                # For now the force is constant and not dependent on my distance to the middle
                B[current_state,right_state,:] += p_o*(1-p_i)
            elif (orientation_index > ori_middle):
                B[current_state,left_state,:] += p_o*(1-p_i)
            else : 
                B[current_state,current_state,:] += p_o*(1-p_i)
            
            # INTENSITY ONLY
            low_state = state_index(intensity_index-1,orientation_index)
            high_state = state_index(intensity_index+1,orientation_index)
            try :
                B[current_state,low_state,:] += p_i*(1-p_o)
            except :
                B[current_state,current_state,:] += p_i*(1-p_o)

            # BOTH INTENSITY AND ORIENTATION
            low_left_state = state_index(intensity_index-1,orientation_index-1)
            high_left_state = state_index(intensity_index-1,orientation_index-1)
            low_right_state = state_index(intensity_index-1,orientation_index-1)
            high_right_state = state_index(intensity_index+1,orientation_index-1)
            if (orientation_index<ori_middle):
                # I'm on the left, decay leads me towards the right
                # For now the force is constant and not dependent on my distance to the middle
                try :
                    B[current_state,low_right_state,:] += p_i*p_o # Go bottom left
                except :
                    B[current_state,right_state,:] += p_i*p_o # Can't go lower, just slide left
            
            elif (orientation_index > ori_middle):
                # I'm on the right, decay leads me towards the left
                # For now the force is constant and not dependent on my distance to the middle
                try :
                    B[current_state,low_left_state,:] += p_i*p_o # Go bottom left
                except :
                    B[current_state,left_state,:] += p_i*p_o # Can't go lower, just slide right
            
            else : # Am on the middle
                try :
                    B[current_state,low_state,:] += p_i*p_o # Just go down ...
                except :
                    B[current_state,current_state,:] += p_i*p_o # ... if you can

            #  "Do not go gentle into that good night,
            #   Old age should burn and rave at close of day;
            #   Rage, rage against the dying of the light." Dylan Thomas
            # --> Our agent can take actions to fight this attentional decay

            # ACTION 1 : STAY HERE (lazy >:( )
            p_action = (1-p_o)*(1-p_i)
            B[current_state,current_state,0] += p_action



    Ns = [3,3]
    D_ = []
    D_.append(np.array([1,1,1])) # Right initial state
    D_.append(np.array([1,1,1])) # Left initial state
    D_ = normalize(D_)

    d_ =[]
    if (perfect_d):
        d_ = D_
        learn_d = False
    else :
        d_.append(np.ones(D_[0].shape))
        d_.append(np.ones(D_[1].shape))
    # Neutral priors about the starting states

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # OBSERVATIONS : 
    # Generally : A[modality] is of shape (Number of outcomes for this modality) x (Number of states for 1st factor) x ... x (Number of states for nth factor)
    # Depending on the LNT or RNT, the A matrix is different :
    
    # Here, let's pick 5 feedback levels possible, equivalent to the
    # difference between the two states
    No = 5 # 1-very_bad 2-bad 3-neutral 4-good 5-very_good
    if SHAM=="False" :
        A_ = np.zeros((No,3,3))

        # Right is at zero :
        if (neurofeedback_training_group =="left"):
            # Left alpha is to be very distinct from right alpha
            # We model this with a simple state correlated to the level 
            # of alpha :
            # If right is low :
            A_[:,0,:] = np.array([[0,0,0],  # Whatever the value of left, not very bad
                                  [0,0,0],  # Whatever the value of left, not  bad
                                  [1,0,0],  # Neutral feedback if left is low too
                                  [0,1,0],  # Good feedback if left is medium
                                  [0,0,1]]) # Very good feedback if left is high

            A_[:,1,:] = np.array([[0,0,0],  # Whatever the value of left, not very bad
                                  [1,0,0],  # If left low and right medium, its bad
                                  [0,1,0],  # Neutral feedback if left is medium too
                                  [0,0,1],  # Good feedback if left is high
                                  [0,0,0]]) # Will never manage a perfect feedback with medium right assymetry

            A_[:,2,:] = np.array([[1,0,0],  # 
                                  [0,1,0],  #
                                  [0,0,1],  # Neutral feedback is a best case scenario
                                  [0,0,0],  # If the level of right alpha is high
                                  [0,0,0]]) # 
            
        elif (neurofeedback_training_group =='right'):
            # Left alpha is to be very distinct from right alpha
            # We model this with a simple state correlated to the level 
            # of alpha :
            # If left is low :
            A_[:,:,0] = np.array([[0,0,0],  # Whatever the value of right, not very bad
                                  [0,0,0],  # Whatever the value of right, not  bad
                                  [1,0,0],  # Neutral feedback if right is low too
                                  [0,1,0],  # Good feedback if right is medium
                                  [0,0,1]]) # Very good feedback if right is high

            A_[:,:,1] = np.array([[0,0,0],  # Whatever the value of right, not very bad
                                  [1,0,0],  # If left low and right medium, its bad
                                  [0,1,0],  # Neutral feedback if right is medium too
                                  [0,0,1],  # Good feedback if right is high
                                  [0,0,0]]) # Will never manage a perfect feedback with medium left assymetry

            A_[:,:,2] = np.array([[1,0,0],  # 
                                  [0,1,0],  #
                                  [0,0,1],  # Neutral feedback is a best case scenario
                                  [0,0,0],  # If the level of left alpha is high
                                  [0,0,0]]) # 
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
    
    increase_activity = np.array([[0,0,0],
                                  [1,0,0],
                                  [0,1,1]])
    decrease_activity = np.array([[1,1,0],
                                  [0,0,1],
                                  [0,0,0]])
    neutral_activity = np.array([[1,0,0],
                                  [0,1,0],
                                  [0,0,1]])
    
    n_b = 8
    B_ = [np.zeros(tuple(Ns)+(n_b,)),np.zeros(tuple(Ns)+(n_b,))]

    for factor in range(Nf):
        B_[factor][:,:,0] = increase_activity
        B_[factor][:,:,1] = decrease_activity
        for k in range(2,n_b):
            B_[factor][:,:,k] = neutral_activity
    
    if (perfect_b):
        b_ = B_
        learn_b = False
    else :
        b_ = [0,0]
        b_[0] = prior_b_confidence*(np.ones(B_[0].shape) + (1-prior_b_precision)*B_[0])
        b_[1] = prior_b_confidence*(np.ones(B_[1].shape) + (1-prior_b_precision)*B_[1])


    number_of_neutral_actions = 4
    nu = 4 + number_of_neutral_actions
    U_ = np.zeros((nu,len(Ns)))
    U_[0,:] = [0,2] # Increase right
    U_[1,:] = [1,3] # Decrease right

    U_[2,:] = [2,0] # Increase left
    U_[3,:] = [3,1] # Decrease left

    for k in range(4,nu):
        U_[k,:] = [k,k] # Neutral action 1
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
    e_ = np.ones((nu,))


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