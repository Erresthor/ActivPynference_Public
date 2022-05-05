from genericpath import isfile
from ipaddress import collapse_addresses
from locale import normalize
from turtle import color
import numpy as np
import os
import matplotlib.pyplot as plt
import imageio,PIL
import statistics as stat


from pyai.layer.layer_learn import MemoryDecayType
from pyai.base.miscellaneous_toolbox import isField
from pyai.base.plotting_toolbox import multi_matrix_plot
from pyai.base.function_toolbox import normalize
from pyai.base.matrix_functions import matrix_distance_list
from pyai.model.active_model import ActiveModel

from pyai.model.active_model_save_manager import ActiveSaveManager

import pyai.model.model_visualizer as vizu
from pyai.model.model_visualizer import load_containers_in_folder,open_model_container

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

    
def generate_model_sumup(modelname,savepath,modality_indice = 0 ,factor_indice = 0,adims=(800,800),bdims=(1500,325,1),colmap = [ np.array([0,0,0,255]) , np.array([95,95,180,255]) , np.array([255,239,10,255]) , np.array([255,100,100,255])]) :
    loadpath = os.path.join(savepath,modelname)
    width,height = adims[0],adims[1]
    bwidth,bheight,lim = bdims[0],bdims[1],bdims[2]
    A_list = []
    B_list = []
    D_list = []
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
            A_list.append([])
            B_list.append([])
            D_list.append([])
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

    
    # SAVING THE RESULTS FOR THE FIRST INSTANCE
    # GIF -->
    result_savepath = os.path.join(savepath,modelname,"_RESULTS")
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
    print(img.shape)
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


def nf_model(modelname,savepath):
    Nf = 1

    initial_state = 0
    D_ =[]
    D_.append(np.array([0,0,0,0,0])) #[Terrible state, neutral state , good state, great state, excessive state]
    #D_[0][initial_state] = 1
    D_ = normalize(D_)

    d_ =[]
    #d_.append(np.array([0.996,0.001,0.001,0.001,0.001])) #[Terrible state, neutral state , good state, great state, excessive state]
    d_.append(np.zeros(D_[0].shape))

    # State Outcome mapping and beliefs
    # Prior probabilities about initial states in the generative process
    Ns = [5] #(Number of states)
    No = [5]

    # Observations : just the states 
    A_ = []

    # Generally : A[modality] is of shape (Number of outcomes for this modality) x (Number of states for 1st factor) x ... x (Number of states for nth factor)
    pa = 1.0
    A_obs_mental = np.array([[pa  ,0.5-0.5*pa,0         ,0         ,0   ],
                            [1-pa,pa        ,0.5-0.5*pa,0         ,0   ],
                            [0   ,0.5-0.5*pa,pa        ,0.5-0.5*pa,0   ],
                            [0   ,0         ,0.5-0.5*pa,pa        ,1-pa],
                            [0   ,0         ,0         ,0.5-0.5*pa,pa  ]])
    A_ = [A_obs_mental]



    prior_ratio = 3 # Correct_weights = ratio*incorrect_weights --> The higher this ratio, the better the quality of the priors
    prior_strength = 2.0 # Base weight --> The higher this number, the stronger priors are and the longer it takes for experience to "drown" them \in [0,+OO[
        
    a_ = []
    a_.append(np.ones((A_[0].shape))*prior_strength)
    a_[0] = a_[0] + (prior_ratio-1.0)*prior_strength*A_[0]
    # for i in range(5):
    #     a_[0][i,i] = 1

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

    b_[0] = 1.0*b_[0] - 0.0*B_[0]

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


    T = 10
    savemanager = ActiveSaveManager(T,trial_savepattern=1,intermediate_savepattern=0)
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
    nf_model.layer_options.T_horizon = 2
    nf_model.layer_options.learn_during_experience = False
    nf_model.layer_options.memory_decay = MemoryDecayType.NO_MEMORY_DECAY #MemoryDecayType.NO_MEMORY_DECAY
    nf_model.layer_options.decay_half_time = 1000000000

    return nf_model


savepath = os.path.join("C:",os.sep,"Users","annic","OneDrive","Bureau","Phd","ActivPynference_Public - New","results","pandas_df")
modelname = "test_free_energy_dataframe"
model = nf_model(modelname,savepath)

L = 1
N = 150
model.initialize_n_layers(L)
#model.run_n_trials(N)

modality_indice = 0
factor_indice = 0

# generate_model_sumup(modelname,savepath,modality_indice,factor_indice)

 
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

def evaluate_container(container,metric='2'):
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
        mean_errors_state += abs(optimal_state-actual_state)/max_size
        
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
    A_mean_distance = stat.mean(matrix_distance_list(container.A_,normalize(container.a_),metric=metric))
    B_mean_distance = stat.mean(matrix_distance_list(container.B_,normalize(container.b_),metric=metric))
    C_mean_distance = stat.mean(matrix_distance_list(container.C_,normalize(container.c_),metric=metric))
    D_mean_distance = stat.mean(matrix_distance_list(container.D_,normalize(container.d_),metric=metric))
    if (isField(container.E_)):
        E_mean_distance = stat.mean(matrix_distance_list(container.E_,normalize(container.e_),metric=metric))
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
    print(free_energy_a,free_energy_b,free_energy_c,free_energy_d,free_energy_e)
    
    
    output_dict = {
        'mean_error_state':mean_errors_state,
        'mean_error_behaviour':mean_error_behaviour,
        'fe_a':free_energy_a,
        'fe_b':free_energy_b,
        'fe_c':free_energy_c,
        'fe_d':free_energy_d,
        'fe_e':free_energy_e,
        'a_dist':A_mean_distance,
        'b_dist':B_mean_distance,
        'c_dist':C_mean_distance,
        'd_dist':D_mean_distance,
        'e_dist':E_mean_distance
    }
    return output_dict
    
cont = open_model_container(os.path.join(savepath,modelname),0,45,'f')
print(evaluate_container(cont))


# #load_containers_in_folder(loadpath)
# cont = open_model_container(loadpath,0,150,9)
# print(cont.a_)

# mod = ActiveModel.load_model(loadpath)
# print(mod.a)

# #vizu.show_figures(mod,cont)

# for file in os.listdir(loadpath):
#     print(file)
#     complete_path = os.path.join(loadpath,file)
#     is_file = (os.path.isfile(complete_path))
#     is_dir = (os.path.isdir(complete_path))
#     if (is_file) :
#         # This is a MODEL file : let's open it
#         model = ActiveModel.load_model(loadpath)
#     if (is_dir) :
#         # This is trial results (layer instance)
#         layer_instance = int(file)
#         for newfile in os.listdir(complete_path):
#             #print(newfile)
#             L = newfile.split("_")
#             trial_counter = int(L[0])
#             timestep_counter = int(L[1])
#             #print(layer_instance,trial_counter,timestep_counter)
#             cont = open_model_container(loadpath,layer_instance,trial_counter,timestep_counter)
#             print(normalize(cont.a_))
#print((cnt.a_[0]*255).astype(int))