from re import S
from tabnanny import verbose
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import colors as mcolors
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from PyAI.pyai.base.file_toolbox import save_flexible
from pyai.model.active_model_save_manager import ActiveSaveManager
from pyai.models_neurofeedback.article_1_simulations.bagherzadeh.orientation_vs_intensity import neurofeedback_model
from scipy.signal import savgol_filter

def return_all_from(fullpath,Ntrials) :
    all_perfs = []
    all_perfo = []
    all_a =[]
    for instance in os.listdir(fullpath):
        if (not("MODEL" in instance)) :
            list_of_perfo = []
            list_of_perfs = []
            list_of_a = []
            for t in range(Ntrials):
                container = ActiveSaveManager.open_trial_container(fullpath,int(instance),t,'f')
                # print(container.o)
                all_s = container.s
                all_o = container.o
                a = container.a_
                # print(avg_obs/4.0)
                list_of_perfs.append(all_s) # rating between 0 and 1
                list_of_perfo.append(all_o)
                list_of_a.append(a)
            all_perfs.append(list_of_perfs)
            all_perfo.append(list_of_perfo)
            all_a.append(list_of_a)
    return all_perfs,all_perfo,all_a

if __name__ == "__main__":
    T = 11

    input_arguments = sys.argv
    assert len(input_arguments)>=2,"Data generator needs at least 4 arguments : savepath and an index + n_instances and n_trials"
    name_of_script = input_arguments[0]
    savepath = input_arguments[1]
    list_index = int(input_arguments[2])
    Ninstances = int(input_arguments[3])
    Ntrials = int(input_arguments[4])
    try :
        overwrite = (input_arguments[5]== 'True')
    except :
        overwrite = False
    
    p_is = np.arange(0,0.5,0.05)
    p_os = np.arange(0,0.5,0.05)
    counter = 0
    p_i = None
    p_o = None
    for i in range(p_is.shape[0]):
        for o in range(p_os.shape[0]):
            if (counter == list_index):
                p_i = p_is[i]
                p_o = p_os[o]
            else :
                counter = counter + 1
    if (p_i==None)or(p_o==None) :
        print("List index out of range ... Aborting instance")
        sys.exit()
    print(p_is,p_os)
    general_name= "Model_"+str(p_i)+"_"+str(p_o)
    modelnameR = os.path.join(general_name,"R")
    save_pathR = savepath
    modelR = neurofeedback_model(modelnameR,save_pathR,p_i,p_o,'right',perfect_a=False,perfect_b = True,perfect_d=True,prior_b_precision=5,prior_a_precision=1,prior_a_confidence=0.5,verbose=True)
    modelR.index = [p_i,p_o,list_index]
    modelR.T = T
    modelR.initialize_n_layers(Ninstances)

    modelnameL = os.path.join(general_name,"L")
    save_pathL = savepath
    modelL = neurofeedback_model(modelnameL,save_pathL,p_i,p_o,'left' ,perfect_a=False,perfect_b = True,perfect_d=True,prior_b_precision=5,prior_a_precision=1,prior_a_confidence=0.5,verbose=True)
    modelL.index = [p_i,p_o,list_index]
    modelL.T = T
    modelL.initialize_n_layers(Ninstances)
    
    modelR.run_n_trials(Ntrials,overwrite=overwrite,global_prop=None)
    modelL.run_n_trials(Ntrials,overwrite=overwrite,global_prop=None)
    
    
    
    
    fullpathR = os.path.join(save_pathR,modelnameR)
    fullpathL = os.path.join(save_pathR,modelnameL)

    s_R, o_R , a_R = return_all_from(fullpathR,Ntrials)
    s_L, o_L , a_L = return_all_from(fullpathL,Ntrials)
    perf_filename = os.path.join(savepath,general_name,"PERFORMANCES.pyai")

    save_this = {
        "modelR" : modelR,
        "modelL" : modelL,
        "s_R" : s_R,
        "s_L" : s_L,
        "o_R" : o_R,
        "o_L" : o_L,
        "a_R" : a_R,
        "a_L" : a_L
    }

    save_flexible(save_this,perf_filename)