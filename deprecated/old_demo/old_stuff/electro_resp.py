from pyai.layer.mdp_layer import mdp_layer
from pyai.layer.electrophysiological_responses import generate_electroph_responses

from pickle import FALSE
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator,FixedLocator
import PIL
import statistics as stat
from pyai.model.metrics import flexible_entropy


from pyai.layer.layer_learn import MemoryDecayType
from pyai.base.miscellaneous_toolbox import isField
from pyai.base.plotting_toolbox import multi_matrix_plot
from pyai.base.function_toolbox import normalize
from pyai.base.matrix_functions import matrix_distance_list,calculate_uncertainty,mean_uncertainty,argmean
from pyai.layer.electrophysiological_responses import generate_electroph_responses

from pyai.model.active_model import ActiveModel
from pyai.model.active_model_save_manager import ActiveSaveManager
from pyai.model.active_model_container import ActiveModelSaveContainer

from pyai.model.model_visualizer import load_containers_in_folder,open_model_container


def nf_model(modelname,savepath,prop_poubelle = 0.0,prior_a_ratio = 3,prior_a_strength=3,prior_b_ratio = 0.0,prior_b_strength=1):
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


    b_ = B_
    # for i in range(B_[0].shape[-1]):
    #     b_[0][:,:,i][B_[0][:,:,i]>0.5] += 10

    No = [A_[0].shape[0]]

    la = -2
    rs = 2
    C_mental = np.array([[2*la],
                        [0.5*la],
                        [0],
                        [0.5*rs],
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


    a_ = A_

    T = 10
    savemanager = ActiveSaveManager(T,trial_savepattern=1,intermediate_savepattern=0)
                                    # Trial related save , timestep related save
    nf_model = ActiveModel(savemanager,modelname,savepath)
    nf_model.T = T
    nf_model.A = A_
    nf_model.a = a_
    nf_model.B = B_
    nf_model.b = b_
    nf_model.C = C_
    nf_model.D = D_
    nf_model.d = d_
    nf_model.U = U_

    nf_model.layer_options.learn_a = False
    nf_model.layer_options.learn_b = False
    nf_model.layer_options.learn_d = True

    nf_model.layer_options.T_horizon = 2
    nf_model.layer_options.learn_during_experience = False
    
    nf_model.layer_options.memory_decay = MemoryDecayType.NO_MEMORY_DECAY
    nf_model.layer_options.memory_decay = MemoryDecayType.STATIC
    nf_model.layer_options.decay_half_time = 200000

    return nf_model

def colorfunc(colorlist,t,interp = 'linear'):
        n = len(colorlist)
        if (interp=='linear'):
            for i in range(n):
                current_color_prop = (float(i)/(n - 1))
                next_color_prop = (float(i+1)/(n-1))
                if ((t>=current_color_prop) and (t<=next_color_prop)):
                    ti = (t - current_color_prop)/(next_color_prop-current_color_prop)
                    return colorlist[i+1]*ti + colorlist[i]*(1-ti)

def custom_colormap(colormap,in_array,interpolation='linear') :
    """Not very elegant + only designed for 3D matrices :>(  """
    output_array = np.zeros(in_array.shape+colormap[0].shape)
    for x in range(in_array.shape[0]):
        for y in range(in_array.shape[1]):
            output_array[x,y,:] = colorfunc(colormap,in_array[x,y],interp=interpolation)
    return output_array
        
def draw_a_3D_image(matrix,intermatrix_size=0,colormap=[np.array([55,0,55,255]),np.array([0,255,255,255])]): # input in [0,1]
    matrix_shape = matrix.shape
    x = matrix_shape[0]
    y = matrix_shape[1]
    z = matrix_shape[2]
       
    pre_y = y*z + intermatrix_size*(z - 1)
    pre_x = x
    
    colsize = colormap[0].shape[0]
    output_array = np.zeros((pre_x,pre_y,colsize))  # RGB / RGBA
    
    low = 0
    high = x
    for zi in range(z):
        expanded_dims_mat = np.zeros((x,y,4))
        expanded_dims_mat[:,:,0] = 255*matrix[:,:,zi]
        expanded_dims_mat[:,:,1] = 0
        expanded_dims_mat[:,:,2] = 255*matrix[:,:,zi]
        expanded_dims_mat[:,:,3] = 255
        expanded_dims_mat = custom_colormap(colormap,matrix[:,:,zi],'linear')
        
        output_array[:,low:high,:] = expanded_dims_mat
    
        if(high<pre_y-1)and(intermatrix_size>0) :
            # draw intermatrix
            output_array[:,high:high+intermatrix_size,:] = 255*np.array([1,1,1,0])
        
        low = high + intermatrix_size
        high = low + x
        
    return  PIL.Image.fromarray(output_array.astype(np.uint8))

def number_of_trials_in_instance_folder(path):
    counter = 0
    for file in os.listdir(path):
        instance = int(file.split("_")[0])
        counter = counter + 1
    return counter
  
def generate_a_dataframe(modelname,savepath,modality_indice = 0 ,factor_indice = 0) :
    loadpath = os.path.join(savepath,modelname)
    
    for file in os.listdir(loadpath):
        print(file)
        complete_path = os.path.join(loadpath,file)
        is_file = (os.path.isfile(complete_path))
        is_dir = (os.path.isdir(complete_path))
        
        if (is_file) :
            # This is a MODEL file : let's open it
            model = ActiveModel.load_model(loadpath)

        if (is_dir) :
            if (file == "_RESULTS"):
                #ignore this file 
                break
            
            # This is trial results (layer instance)
            layer_instance = int(file)
            for newfile in os.listdir(complete_path):
                L = newfile.split("_")
                trial_counter = int(L[0])
                timestep_counter = int(L[1])
                cont = open_model_container(loadpath,layer_instance,trial_counter,timestep_counter)
                
                
                # A is matrix of dimensions outcomes x num_factors_for_state_factor_0 x num_factors_for_state_factor_1 x ... x num_factors_for_state_factor_n
                # draw a 3D image is not made for matrix with dimensions >= 4. As a general rule, we take the initial dimensions of the matrix and pick the 0th 
                # Indices of excess dimensions :
                print(cont.as_dict())
                print(cont.return_dataframe())

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
    
    # KL dirs (calculated in the respective free energies :D)
    free_energy_a = container.FE['Fa']
    free_energy_b = container.FE['Fb']
    free_energy_c = container.FE['Fc']
    free_energy_d = container.FE['Fd']
    free_energy_e = container.FE['Fe']
    #print(free_energy_a,free_energy_b,free_energy_c,free_energy_d,free_energy_e)
    
    factor = 0.5
    try :
        mean_uncertainty_a = mean_uncertainty(container.a_,factor)
        mean_uncertainty_a = flexible_entropy(container.a_)
    except :
        mean_uncertainty_a = [0 for i in range(len(container.A_))]
    try :
        mean_uncertainty_b = mean_uncertainty(container.b_,factor)
        mean_uncertainty_b = flexible_entropy(container.b_)
    except :
        mean_uncertainty_b = [0 for i in range(len(container.B_))]
    try :
        mean_uncertainty_d = mean_uncertainty(container.d_,factor)
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
        'a_uncertainty': mean_uncertainty_a,
        'b_uncertainty': mean_uncertainty_b,
        'd_uncertainty': mean_uncertainty_d
    }
    return output_dict    
  
def generate_model_sumup_old(modelname,savepath,modality_indice = 0 ,factor_indice = 0,adims=(800,800),bdims=(1500,325,1),colmap = [ np.array([0,0,0,255]) , np.array([95,95,180,255]) , np.array([255,239,10,255]) , np.array([255,100,100,255])]) :
    """Generate the sumup figures for a single model layer instance accross all layer instance.
          /----> inst 1 ---> sumup 1
    MODEL -----> inst 2 ---> sumup 2
          \----> inst 3 ---> sumup 3
    """
    
    loadpath = os.path.join(savepath,modelname)
    width,height = adims[0],adims[1]
    bwidth,bheight,lim = bdims[0],bdims[1],bdims[2]
    A_list = []
    B_list = []
    D_list = []
    
    model = ActiveModel.load_model(loadpath)

    for file in os.listdir(loadpath):
        print("Opening file  : "+ file)
        complete_path = os.path.join(loadpath,file)
        is_file = (os.path.isfile(complete_path))
        is_dir = (os.path.isdir(complete_path))
        
        

        if (is_dir) :
            if ("_RESULTS" in file) or ("_MODEL" in file):
                #ignore this file 
                continue
            A_list.append([])
            B_list.append([])
            D_list.append([])
            # This is trial results (layer instance)
            layer_instance = int(file)

            listdir_counter = 0
            HARDLIMIT = 1000
            len_dir = len(os.listdir(complete_path))
            for newfile in os.listdir(complete_path):
                L = newfile.split("_")
                trial_counter = int(L[0])
                timestep_counter = 'f'
                cont = open_model_container(loadpath,layer_instance,trial_counter,timestep_counter)
                
                
                # A is matrix of dimensions outcomes x num_factors_for_state_factor_0 x num_factors_for_state_factor_1 x ... x num_factors_for_state_factor_n
                # draw a 3D image is not made for matrix with dimensions >= 4. As a general rule, we take the initial dimensions of the matrix and pick the 0th 
                # Indices of excess dimensions :
                
                if (listdir_counter < HARDLIMIT)or(listdir_counter==len_dir-1): # Either get the HARDLIMIT firsts or the last one
                
                    try :
                        while (cont.a_[modality_indice].ndim > 3):
                            cont.a_[modality_indice] = cont.a_[modality_indice][...,0]
                        a_image = draw_a_3D_image(normalize(np.expand_dims(cont.a_[0],-1)), lim,colormap =colmap)
                    except :
                        while (cont.A_[modality_indice].ndim > 3):
                            cont.A_[modality_indice] = cont.a_[modality_indice][...,0]
                        a_image = draw_a_3D_image(normalize(np.expand_dims(cont.A_[0],-1)), lim,colormap =colmap)
                    a_resized = a_image.resize((width,height),PIL.Image.Resampling.NEAREST)
                    A_list[-1].append(a_resized)
                    

                    try :
                        b_image = draw_a_3D_image(normalize(cont.b_[factor_indice]),lim,colormap=colmap)
                    except :
                        b_image = draw_a_3D_image(normalize(cont.B_[factor_indice]),lim,colormap=colmap)
                    b_resized = b_image.resize((bwidth,bheight),PIL.Image.Resampling.NEAREST)
                    B_list[-1].append(b_resized)
                    
                    
                    try :
                        d_image = draw_a_3D_image(np.expand_dims(np.expand_dims(normalize(cont.d_[factor_indice]),-1),-1), lim,colormap =colmap)
                    except :
                        d_image = draw_a_3D_image(np.expand_dims(np.expand_dims(normalize(cont.D_[factor_indice]),-1),-1), lim,colormap =colmap)
                    d_resized = d_image.resize((width,height),PIL.Image.Resampling.NEAREST)
                    D_list[-1].append(d_resized)

    
            # SAVING THE RESULTS FOR THE NTH INSTANCE
            # GIF -->
            instance_str = file
            result_savepath = os.path.join(savepath,modelname,"_RESULTS_"+instance_str)
            if not os.path.exists(result_savepath):
                try:
                    os.makedirs(result_savepath)
                except OSError as exc: # Guard against race condition
                    raise

            fi = min(75,len(B_list[0]))  # The first frames shown on a slower pace to get better understanding of learning dynamics
            
            savepath_gif = os.path.join(result_savepath,"b__" + str(modelname) + ".gif")
            B_list[0][0].save(savepath_gif,append_images=B_list[0][1:],save_all=True,duration=30,loop=0)
            savepath_gif = os.path.join(result_savepath,"first"+str(fi)+"_b__" + str(modelname) + ".gif")
            B_list[0][0].save(savepath_gif,append_images=B_list[0][1:fi],save_all=True,duration=150,loop=0)

            savepath_gif = os.path.join(result_savepath,"a__" + str(modelname) + ".gif")
            A_list[0][0].save(savepath_gif, format = 'GIF',append_images=A_list[0][1:],save_all=True,duration=30,loop=0)
            savepath_gif = os.path.join(result_savepath,"first"+str(fi)+"_a__" + str(modelname) + ".gif")
            A_list[0][0].save(savepath_gif, format = 'GIF',append_images=A_list[0][1:fi],save_all=True,duration=150,loop=0)


            # Save final results for the first instance :
            savepath_img = os.path.join(result_savepath,"first_a__" + str(modelname) + ".png")
            A_list[0][0].save(savepath_img)
            savepath_img = os.path.join(result_savepath,"final_a__" + str(modelname) + ".png")
            A_list[0][-1].save(savepath_img)
            savepath_img = os.path.join(result_savepath,"first_b__" + str(modelname) + ".png")
            B_list[0][0].save(savepath_img)
            savepath_img = os.path.join(result_savepath,"final_b__" + str(modelname) + ".png")
            B_list[0][-1].save(savepath_img)

            #Save scale for the first instance :
            savepath_img = os.path.join(result_savepath,"colorscale__" + str(modelname) + ".png")
            N = 500
            img_array = np.linspace(0,1,N)
            img = np.zeros((100,) + img_array.shape + (4,))
            for k in range(N):
                color_array = colorfunc(colmap,img_array[k])
                img[:,k,:] = color_array
            img = PIL.Image.fromarray(img.astype(np.uint8))
            img.resize((800,100))
            img.save(savepath_img)


            B = model.B[factor_indice]
            try :
                b = model.b[factor_indice]
                b_ = cont.b_[factor_indice]
            except :
                b = B
                b_ = B

            A = model.A[modality_indice]
            try :
                a = model.a[modality_indice]
                a_ = cont.a_[modality_indice]
            except :
                a = A
                a_ = A
                
            D = model.D[factor_indice]
            try :
                d = model.d[factor_indice]
                d_ = cont.d_[factor_indice]
            except :
                d = D
                d_ = D

            multi_matrix_plot([normalize(B),normalize(b),normalize(b_)],["Real B","Prior b","Learnt b"],"FROM states","TO states")
            savepath_img = os.path.join(result_savepath,"sumup_B__" + str(modelname) + ".png")
            plt.savefig(savepath_img,bbox_inches='tight',dpi=1000)
            plt.close()

            multi_matrix_plot([normalize(A),normalize(a),normalize(a_)], ["Real A","Prior a","Learnt a"],"State (cause)","Observation (consequence)")
            savepath_img = os.path.join(result_savepath,"sumup_A__" + str(modelname) + ".png")
            plt.savefig(savepath_img,bbox_inches='tight',dpi=1000)
            plt.close()

            multi_matrix_plot([normalize(D),normalize(d),normalize(d_)], ["Real D","Prior d","Learnt d"], "Initial belief","State")
            savepath_img = os.path.join(result_savepath,"sumup_D__" + str(modelname) + ".png")
            plt.savefig(savepath_img,bbox_inches='tight',dpi=1000)
            plt.close()
 
def generate_model_sumup(modelname,savepath,gifs=False,modality_indice = 0 ,factor_indice = 0,adims=(800,800),bdims=(1500,325,1),colmap = [ np.array([0,0,0,255]) , np.array([95,95,180,255]) , np.array([255,239,10,255]) , np.array([255,100,100,255])]) :
    """Generate the sumup figures for a single model layer instance accross all layer instance.
          /----> inst 1 ---> sumup 1
    MODEL -----> inst 2 ---> sumup 2
          \----> inst 3 ---> sumup 3
    """
    loadpath = os.path.join(savepath,modelname)
    width,height = adims[0],adims[1]
    bwidth,bheight,lim = bdims[0],bdims[1],bdims[2]
    A_list = []
    B_list = []
    D_list = []
    
    model = ActiveModel.load_model(loadpath)
    for file in os.listdir(loadpath):
        
        complete_path = os.path.join(loadpath,file)
        is_file = (os.path.isfile(complete_path))
        is_dir = (os.path.isdir(complete_path))

        if (is_dir) :

            if ("_RESULTS" in file) or ("_MODEL" in file):
                print("Ignoring  file : "+ file)
                #ignore this file 
                continue
            print("Generating sumup for file  : "+ file)
            A_list.append([])
            B_list.append([])
            D_list.append([])
            # This is trial results (layer instance)
            layer_instance = int(file)

            listdir_counter = 0
            HARDLIMIT = 1000
            len_dir = len(os.listdir(complete_path))
            for newfile in os.listdir(complete_path):
                L = newfile.split("_")
                trial_counter = int(L[0])
                timestep_counter = 'f'
                cont = open_model_container(loadpath,layer_instance,trial_counter,timestep_counter)
                
                # A is matrix of dimensions outcomes x num_factors_for_state_factor_0 x num_factors_for_state_factor_1 x ... x num_factors_for_state_factor_n
                # draw a 3D image is not made for matrix with dimensions >= 4. As a general rule, we take the initial dimensions of the matrix and pick the 0th 
                # Indices of excess dimensions :
                
                if (listdir_counter < HARDLIMIT)or(listdir_counter==len_dir-1): # Either get the HARDLIMIT firsts or the last one
                
                    try :
                        a_image = cont.a_
                    except :
                        a_image = cont.A_
                    A_list[-1].append(a_image)

                    try :
                        b_image = cont.b_
                    except :
                        b_image = cont.B_
                    B_list[-1].append(b_image)
                    
                    
                    try :
                        d_image = cont.d_
                    except :
                        d_image = cont.D_
                    D_list[-1].append(d_image)

            belief_matrices_plot(modelname,savepath,A_list[-1],B_list[-1],D_list[-1],
                    adims=adims,bdims=bdims,
                    colmap = colmap,
                    plot_modality=modality_indice,plot_factor = factor_indice,instance_string=file,plot_gifs=gifs)

def general_performance_sumup(savepath,modelname,instance_number=0):
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
        cont = open_model_container(os.path.join(savepath,modelname),instance_number,trial,'f')
        eval_cont = evaluate_container(cont)
        trials.append(trial) 
        a_err.append(eval_cont['a_dist'])
        b_err.append(eval_cont['b_dist'])
        Ka = Ka + eval_cont['a_uncertainty']
        Kb = Kb + eval_cont['b_uncertainty']
        Kd = Kd + eval_cont['d_uncertainty']

        error_states.append(evaluate_container(cont)['mean_error_state'])
        error_behaviour.append(evaluate_container(cont)['mean_error_behaviour'])
    return trials,a_err,b_err,Ka,Kb,Kd,error_states,error_behaviour
    
def general_performance_figure (savepath,modelname,save_string,trials,a_err,b_err,a_unc,b_unc,error_states,error_behaviour,smooth_window = 5,show=True,asp_ratio=(10,5),
                                figtitle = "untitled") :
    # Mean of error states and behaviour :
    def sliding_window_mean(list_input,window_size = 3):
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

    state_error_mean = sliding_window_mean(error_states,smooth_window)
    behaviour_error_mean = sliding_window_mean(error_behaviour,smooth_window)


    color1 = 'tab:red'
    color2 = 'tab:blue'
    fig = plt.figure(figsize=asp_ratio)
    ax1 = fig.add_subplot(211)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xlabel('trial')
    ax1.set_ylabel('mean metric entropy', color=color1)
    ax1.set_ylim([-0.1,1.1])
    

    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylabel('mean square error', color=color2)  # we already handled the x-label with ax1
    

    l1 = ax1.plot(trials, a_unc, color=color1,label='A entropy',ls='--')
    l2 = ax1.plot(trials, b_unc, color=color1,label='B entropy',ls='-')
    # instantiate a second axes that shares the same x-axis
    
    l3 = ax2.plot(trials, a_err, color=color2,label='A error',ls='--')
    l4 = ax2.plot(trials, b_err, color=color2,label='B error',ls='-')

    ls = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in ls]
    ax1.legend(ls,labs,loc = 0)

    ax1.grid()
    

    # -----------------------------------------------------------------
    color3 = 'yellow'
    color3l = 'orange'
    color4 = 'cyan'
    color4l = 'purple'
    ax3 = fig.add_subplot(212)
    ax3.grid()

    l1 = ax3.plot(trials,error_states,'*',color=color3,label = 'error w.r.t. optimal states')

    ax4 = ax3.twinx()
    l2 = ax4.plot(trials,error_behaviour,'+',color=color4,label = 'error w.r.t. optimal behaviour')
    l3 = ax3.plot(trials,state_error_mean,"-",color=color3l,label = 'error w.r.t. optimal states (smoothed)')
    l4 = ax4.plot(trials,behaviour_error_mean,"--",color=color4l,label = 'error w.r.t. optimal behaviour (smoothed)')

    ls = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in ls]
    ax3.legend(ls,labs,loc = 'best')

    ax3.set_xlabel('trial')
    ax3.set_ylabel('state error', color=color3l)
    ax3.tick_params(axis='y', labelcolor=color3l)
    ax3.set_ylim([-0.1,1.1])

    ax4.set_ylabel('behaviour error', color=color4l)
    ax4.tick_params(axis='y', labelcolor=color4l)
    ax4.set_ylim([-0.1,1.1])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.suptitle(figtitle, fontsize=16,y=1.08)

    save_folder = os.path.join(savepath,modelname,"_RESULTS_" + save_string)
    

    if not os.path.exists(save_folder):
                try:
                    os.makedirs(save_folder)
                except OSError as exc: # Guard against race condition
                    raise
    
    figname = os.path.join(save_folder,"performances")
    plt.savefig(figname,bbox_inches='tight',dpi=1000)
    if(show):
        plt.show()
    else :
        plt.close()

def general_performance(savepath,modelname,instance_number=0) :
    trials,a_err,b_err,Ka,Kb,Kd,error_states,error_behaviour = general_performance_sumup(savepath,modelname,instance_number)
    save_string = f'{instance_number:03d}'
    figtitle = modelname +" - Instance " + str(instance_number) + " performance sumup"
    general_performance_figure(savepath,modelname,save_string,trials,a_err,b_err,Ka,Kb,error_states,error_behaviour,smooth_window = 5,show=False,figtitle=figtitle)

def belief_matrices_plot(modelname,savepath,a_list,b_list,d_list,
                    adims=(800,800),bdims=(1500,325,1),
                    colmap = [ np.array([0,0,0,255]) , np.array([95,95,180,255]) , np.array([255,239,10,255]) , np.array([255,100,100,255])],
                    plot_modality=0,plot_factor = 0,instance_string = "GLOBAL",plot_gifs=False) :
    """Generate the sumup figures for a mean of all model layer instance
          /----> inst 1 ---\ 
    MODEL -----> inst 2 -----> mean results ----> general sumup figure
          \----> inst 3 ---/                 /|\
                                              |
                                              |
                                              |
                                              |
                                         This is me !
    """
    loadpath = os.path.join(savepath,modelname)

    width,height = adims[0],adims[1]
    bwidth,bheight,lim = bdims[0],bdims[1],bdims[2]
    
    model = ActiveModel.load_model(loadpath) # Get the model that caused this global result

    HARDLIMIT = 500
    len_dir = len(a_list)
    listdir_counter = 0

    a_im_list = []
    b_im_list = []
    d_im_list = []
    for trial in range(len(a_list)):

        if (listdir_counter < HARDLIMIT)or(listdir_counter==len_dir-1): # Either get the HARDLIMIT firsts or the last one
            listdir_counter += 1

            try :
                while (a_list[trial][plot_modality].ndim > 3):
                    a_list[trial][plot_modality] = a_list[trial][plot_modality][...,0]
                a_mat = a_list[trial][plot_modality]
            except :
                while (model.A_[plot_modality].ndim > 3):
                    model.A_[plot_modality] = model.A_[plot_modality][...,0]
                a_mat = model.A_[plot_modality]
            

            if (a_mat.ndim < 3):
                a_mat = np.expand_dims(a_mat,-1)
            a_image = draw_a_3D_image(normalize(a_mat), lim,colormap =colmap)
            a_resized = a_image.resize((width,height),PIL.Image.Resampling.NEAREST)

            a_im_list.append(a_resized)
            

            try :
                b_image = draw_a_3D_image(normalize(b_list[trial][plot_factor]),lim,colormap=colmap)
            except :
                b_image = draw_a_3D_image(normalize(model.B_[plot_factor]),lim,colormap=colmap)
            b_resized = b_image.resize((bwidth,bheight),PIL.Image.Resampling.NEAREST)
            b_im_list.append(b_resized)
            
            
            try :
                d_image = draw_a_3D_image(np.expand_dims(np.expand_dims(normalize(d_list[trial][plot_factor]),-1),-1), lim,colormap =colmap)
            except :
                d_image = draw_a_3D_image(np.expand_dims(np.expand_dims(normalize(model.D_[plot_factor]),-1),-1), lim,colormap =colmap)
            d_resized = d_image.resize((width,height),PIL.Image.Resampling.NEAREST)
            d_im_list.append(d_resized)


    # SAVING THE RESULTS FOR THE MEAN OF ALL INSTANCES
    # GIF -->
    result_savepath = os.path.join(savepath,modelname,"_RESULTS_"+instance_string)
    if not os.path.exists(result_savepath):
        try:
            os.makedirs(result_savepath)
        except OSError as exc: # Guard against race condition
            raise
        
    if (plot_gifs):
        # GIFS : 
        fi = min(75,len(b_im_list))  # The first frames shown on a slower pace to get better understanding of learning dynamics
        
        savepath_gif = os.path.join(result_savepath,"b__" + str(modelname) + ".gif")
        b_im_list[0].save(savepath_gif,append_images=b_im_list[1:],save_all=True,duration=30,loop=0)
        savepath_gif = os.path.join(result_savepath,"b_" + "first"+str(fi)+"__" + str(modelname) + ".gif")
        b_im_list[0].save(savepath_gif,append_images=b_im_list[1:fi],save_all=True,duration=150,loop=0)

        savepath_gif = os.path.join(result_savepath,"a__" + str(modelname) + ".gif")
        a_im_list[0].save(savepath_gif, format = 'GIF',append_images=a_im_list[1:],save_all=True,duration=30,loop=0)
        savepath_gif = os.path.join(result_savepath,"a_" +"first"+str(fi)+ "__" + str(modelname) + ".gif")
        a_im_list[0].save(savepath_gif, format = 'GIF',append_images=a_im_list[1:fi],save_all=True,duration=150,loop=0)


    # Save final results for the first instance :
    savepath_img = os.path.join(result_savepath,"a_first__" + str(modelname) + ".png")
    a_im_list[0].save(savepath_img)
    savepath_img = os.path.join(result_savepath,"a_final__" + str(modelname) + ".png")
    a_im_list[-1].save(savepath_img)

    # Grab the ground truth perception matrix and make it an image
    while (model.A[plot_modality].ndim > 3):
        model.A[plot_modality] = model.A[plot_modality][...,0]
    a_mat = model.A[plot_modality]
    if (a_mat.ndim < 3):
        a_mat = np.expand_dims(a_mat,-1)
    a_image = draw_a_3D_image(normalize(a_mat), lim,colormap =colmap)
    a_resized = a_image.resize((width,height),PIL.Image.Resampling.NEAREST)
    savepath_img = os.path.join(result_savepath,"a_true__" + str(modelname) + ".png")
    a_resized.save(savepath_img)
    
    savepath_img = os.path.join(result_savepath,"b_first__" + str(modelname) + ".png")
    b_im_list[0].save(savepath_img)
    savepath_img = os.path.join(result_savepath,"b_final__" + str(modelname) + ".png")
    b_im_list[-1].save(savepath_img)

    # Grab the ground truth action matrix and make it an image
    b_image = draw_a_3D_image(normalize(model.B[plot_factor]),lim,colormap=colmap)
    b_resized = b_image.resize((bwidth,bheight),PIL.Image.Resampling.NEAREST)
    savepath_img = os.path.join(result_savepath,"b_true__" + str(modelname) + ".png")
    b_resized.save(savepath_img)

    #Save scale for the first instance :
    savepath_img = os.path.join(result_savepath,"zz_colorscale__" + str(modelname) + ".png")
    N = 500
    img_array = np.linspace(0,1,N)
    img = np.zeros((100,) + img_array.shape + (4,))
    for k in range(N):
        color_array = colorfunc(colmap,img_array[k])
        img[:,k,:] = color_array
    img = PIL.Image.fromarray(img.astype(np.uint8))
    img.resize((800,100))
    img.save(savepath_img)


    B = model.B[plot_factor]
    try :
        b = model.b[plot_factor]
        b_ = b_list[-1][plot_factor]
    except :
        b = B
        b_ = B

    A = model.A[plot_modality]
    try :
        a = model.a[plot_modality]
        a_ = a_list[-1][plot_modality]
    except :
        a = A
        a_ = A
        
    D = model.D[plot_factor]
    try :
        d = model.d[plot_factor]
        d_ = d_list[-1][plot_factor]
    except :
        d = D
        d_ = D
    DPI = 150

    multi_matrix_plot([normalize(B),normalize(b),normalize(b_)],["Real B","Prior b","Learnt b"],"FROM states","TO states")
    savepath_img = os.path.join(result_savepath,"B_sumup__" + str(modelname) + ".png")
    plt.savefig(savepath_img,bbox_inches='tight',dpi=DPI)
    plt.close()

    multi_matrix_plot([normalize(A),normalize(a),normalize(a_)], ["Real A","Prior a","Learnt a"],"State (cause)","Observation (consequence)")
    savepath_img = os.path.join(result_savepath,"A_sumup__" + str(modelname) + ".png")
    plt.savefig(savepath_img,bbox_inches='tight',dpi=DPI)
    plt.close()

    multi_matrix_plot([normalize(D),normalize(d),normalize(d_)], ["Real D","Prior d","Learnt d"], "Initial belief","State")
    savepath_img = os.path.join(result_savepath,"D_sumup__" + str(modelname) + ".png")
    plt.savefig(savepath_img,bbox_inches='tight',dpi=DPI)
    plt.close()

def all_indicators(modelname,savepath) : 
    """Return all the perforamnce indicators implemented for a given model accross all layer instances"""
    
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
                cont = open_model_container(loadpath,layer_instance,trial_counter,timestep_counter)
                
                
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
                a_err[-1].append(eval_cont['a_dist'])
                b_err[-1].append(eval_cont['b_dist'])
                Ka[-1] = Ka[-1] + eval_cont['a_uncertainty']
                Kb[-1] = Kb[-1] + eval_cont['b_uncertainty']
                Kd[-1] = Kd[-1] + eval_cont['d_uncertainty']
                error_states[-1].append(evaluate_container(cont)['mean_error_state'])
                error_behaviour[-1].append(evaluate_container(cont)['mean_error_behaviour'])
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

def trial_plot_new(plotfile,plotmean=False,action_labels="alphabet",title=None):
    labelfont = {
        'weight': 'light',
        'size': 8,
        }


    hidden_state_factor = 0
    perc_modality = 0

    cont = ActiveModelSaveContainer.load_active_model_container(plotfile)
    eval_cont = evaluate_container(cont)

    T = cont.T
    timesteps = np.linspace(0,T-1,T)
    
    obs = cont.o
    states = cont.s
    acts = cont.u
    beliefs = cont.X
    print("#############")
    for t in range(T-1):
        print(str(np.round(cont.U_post[:,t],2)) + " -> " + str(acts[hidden_state_factor,t]))
    
    Nactions = cont.U_post.shape[0]
    Ns = beliefs[hidden_state_factor].shape[0]
    No = Ns # In the case of observation-hidden state same size spaces


    my_colormap= [np.array([80,80,80,200]) , np.array([39,136,245,200]) , np.array([132,245,39,200]) , np.array([245,169,39,200]) , np.array([255,35,35,200])]
    
    
    state_belief_image = custom_colormap(my_colormap,beliefs[hidden_state_factor])
    mean_beliefs = argmean(beliefs[hidden_state_factor],axis=0)
            # Only pertinent if states of close indices are spacially 
            # linked    

    # Major ticks every 5, minor ticks every 1
    minor_ticks_x = np.arange(0, T, 1)
    major_ticks_x = np.arange(0, T, 1)-0.5
    ticks_actions = np.arange(0, Nactions, 1)





    # BEGIN ! --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    fig = plt.figure(constrained_layout=True)
    
    subfigures = fig.subfigures(1,2,wspace=0.07, width_ratios=[1.7, 1.])

    

    axes = subfigures[0].subplots(2,1)
    ax1 = axes[0]
    ax2 = axes[1]

    # ax3 = fig.add_subplot(111, zorder=-1)
    # for _, spine in ax3.spines.items():
    #     spine.set_visible(False)
    # ax3.tick_params(labelleft=False, labelbottom=False, left=False, right=False )
    # ax3.get_shared_x_axes().join(ax3,ax1)
    # ax3.set_xticks(minor_ticks,major_ticks)
    # minor_locator1 = AutoMinorLocator(2)
    # ax3.xaxis.set_minor_locator(minor_locator1)
    # ax3.grid(which='minor')

    labels = [str("") for i in minor_ticks_x]
    ax1.set_xticks(minor_ticks_x,major_ticks_x)
    minor_locator1 = AutoMinorLocator(2)
    ax1.xaxis.set_minor_locator(minor_locator1)
    ax1.grid(which='minor')
    ax1.set_xticklabels(labels)

    labels = [("t="+str(i)) for i in minor_ticks_x]
    ax2.set_xticks(major_ticks_x,minor=True)
    ax2.set_xticks(minor_ticks_x)
    ax2.set_yticks(ticks_actions)
    ax2.xaxis.grid(True, which='minor')
    ax2.set_xticklabels(labels)
    
    if (action_labels=="alphabet"):
        letters_list =  list(map(chr,range(ord('a'),ord('z')+1)))
        mylabel = letters_list[:Nactions]
        ax2.set_yticklabels(mylabel)
    
    ax1.set(xlim=(0-0.5, T-0.5))
    ax1.imshow(state_belief_image/255.0,aspect="auto")
    ax1.plot(timesteps,states[hidden_state_factor,:],color='black',lw=3)
    if (plotmean):
        ax1.plot(timesteps,mean_beliefs,color='blue',lw=2,ls=":")
    ax1.plot(timesteps,obs[hidden_state_factor,:],color='w',marker="H",linestyle = 'None',markersize=10)
    ax1.set_ylabel("OBSERVATIONS AND PERCEPTION")
    
    
    ax2.set(xlim=(0-0.5, T-0.5))
    action_posterior_image = custom_colormap(my_colormap,cont.U_post)
    ax2.imshow(action_posterior_image/255.0,aspect="auto")
    ax2.plot(timesteps[:-1],acts[hidden_state_factor,:],color='green',marker="*",linestyle = 'None',markersize=10)
    ax2.set_ylabel("ACTIONS")
    ax2.set_xlabel("Timesteps")
    


    for ax in [ax1,ax2]:
        ax.set_anchor('W')
    # fig.tight_layout()
    # fig.show()


    # fig,axes = plt.subplots(2,1)
    subfigs_nested = subfigures[1].subfigures(2,1)
    axes = subfigs_nested[0].subplots(1,2)
    ax4 = axes[0]
    ax6 = axes[1]
    
    
    a_mat = cont.a_[perc_modality]
    while (a_mat.ndim < 3):
        a_mat = np.expand_dims(a_mat,-1)
    a_image = draw_a_3D_image(normalize(a_mat),colormap =my_colormap)
    ax4.imshow(a_image)
    ax4.set_xlabel("States at time t",font=labelfont)
    ax4.set_ylabel("Cause observations at time t",font=labelfont)
    ax4.set_title('Perception model (after learning)', fontsize=10)

    #Save scale for the first instance :
    N = 250
    img_array = np.linspace(1,0,N)
    img = np.zeros(img_array.shape +(50,) +  (4,))
    for k in range(N):
        color_array = colorfunc(my_colormap,img_array[k])
        img[k,:,:] = color_array
    img = PIL.Image.fromarray(img.astype(np.uint8))
    #img.resize((800,100))
    ax6.imshow(img) 
    ax6.set_title('Color legend', fontsize=10)
    ax6.set_ylabel("Probability density",font=labelfont)
    ax6.set_xticks([])
    ax6.set_yticks([0,N/2.0,N])
    ax6.set_yticklabels(["1.0","0.5","0.0"])

    axes = subfigs_nested[1].subplots(1,1)
    ax5 = axes
    print(cont.b_)
    b_mat = cont.b_[hidden_state_factor]
    lim = 0
    b_image = draw_a_3D_image(normalize(b_mat),lim,colormap =my_colormap)
    
    y_ticks = np.arange(0,Ns,1)
    major_ticks = np.arange(0,Nactions,1)*(Ns+lim)-lim + (Ns+lim)/2.0 -0.5
    minor_ticks=[]
    major_ticks = []
    iamhere=0
    for k in range(Nactions):
        major_ticks.append(iamhere-0.5 + Ns/2.0)
        minor_ticks.append(iamhere-0.5)
        iamhere = iamhere + Ns
        minor_ticks.append(iamhere-0.5)
        iamhere = iamhere + lim
    
    labels = [("t="+str(i)) for i in minor_ticks_x]
    ax5.set_xticks(minor_ticks,minor=True)
    ax5.set_xticks(major_ticks)
    ax5.set_yticks(y_ticks)
    ax5.xaxis.grid(True, which='minor')
    if (action_labels=="alphabet"):
        letters_list =  list(map(chr,range(ord('a'),ord('z')+1)))
        mylabel = letters_list[:Nactions]
        print(mylabel)
        ax5.set_xticklabels(mylabel)
    ax5.set_yticklabels(["" for i in range(Ns)])

    ax5.imshow(b_image)
    ax5.set_title('Action model (after learning)', fontsize=10)
    ax5.set_xlabel("Action X leads from states t",font=labelfont)
    ax5.set_ylabel("To states t+1",font=labelfont)

    
    subfigures[0].suptitle('TRIAL HISTORY', fontsize=13)
    subfigures[1].suptitle('SUBJECT MODEL', fontsize=13)
    if (title==None):
        fig.suptitle('Trial sum-up', fontsize='xx-large')
    else :
        fig.suptitle(title, fontsize='xx-large')
    fig.show()

def generate_figures(savepath,modelname,instance_list,gifs=False,mod_ind=0,fac_ind=0):
    generate_model_sumup(modelname,savepath,gifs,mod_ind,fac_ind)
    for inst in instance_list:
        general_performance(savepath,modelname,inst)

def report_mean_behaviour(savepath,modelname,show=True):
    mean_A,mean_B,mean_D,a_err,b_err,Ka_arr,Kb_arr,Kd_arr,error_states_arr,error_behaviour_arr,tot_instances = mean_indicators_model(modelname,savepath)
    n = a_err.shape[0]
    trials = np.linspace(0,n,n)
    general_performance_figure(savepath,modelname,"GLOBAL",trials,a_err,b_err,Ka_arr,Kb_arr,error_states_arr,error_behaviour_arr,smooth_window=5,figtitle=modelname+" - performance sumup over " + str(tot_instances) + " instance(s)",show=True)
    belief_matrices_plot(modelname,savepath,mean_A,mean_B,mean_D,plot_gifs=True)


# ENVIRONMENT
savepath = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","code","results","2022_06_07")
modelname = "test_dopamine"

# SIMULATING TRAINING
model = nf_model(modelname,savepath,prop_poubelle=0.3,prior_a_ratio=1,prior_a_strength=1,prior_b_ratio=5,prior_b_strength=3)
Ninstances = 1

trials_per_instances = 10
model.initialize_n_layers(Ninstances)

overwrite = True
model.run_n_trials(trials_per_instances,overwrite=overwrite)

k,j = generate_electroph_responses(model.layer_list[0])


model_folder = os.path.join(savepath,modelname)
for trial in [trials_per_instances-1] :
    inst = 0
    full_file_name = ActiveSaveManager.generate_save_name(model_folder,inst,trial,'f')
    trial_plot_new(full_file_name,title="Trial " + str(trial) + " sum-up (instance " + str(inst) + " )")

fig = plt.figure()
ax1 = fig.add_subplot(111)
# print(k.shape,j.shape)
# print(k)
ax1.plot(np.arange(0,k.shape[0],1),k)
ax1.plot(np.arange(0,j.shape[0],1),j)
fig.show()

input()