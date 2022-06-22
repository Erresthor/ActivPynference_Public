from pickle import FALSE
import numpy as np
import os
import matplotlib.pyplot as plt

from pyai.base.file_toolbox import save_flexible,load_flexible

from pyai.layer.layer_learn import MemoryDecayType

from pyai.model.active_model import ActiveModel
from pyai.model.active_model_save_manager import ActiveSaveManager
from pyai.model.active_model_container import ActiveModelSaveContainer

from pyai.neurofeedback_run import run_models

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

def generate_a_dictionnary(a_priors,b_priors) :
    new_dict = {}
    for ka in range(a_priors.shape[0]):
        for kb in range(b_priors.shape[0]):
            modelchar = [True,a_priors[ka],1,True,b_priors[kb],1,True,MemoryDecayType.NO_MEMORY_DECAY,2000]
            modelname = "a_ac"+str(int(10*a_priors[ka]))+"_str1_b_ac"+str(int(10*b_priors[kb]))+"_str1"
            new_dict[modelname] = modelchar
    return new_dict


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


prior_value_a = np.array([1.0,1.2,1.5,1.8,2.0,2.4,2.8,3.0,5.0,15.0,50.0,200.0])
#prior_value_a = np.array([1.0,1.2,1.5,2.0,5.0,15.0])
models_dictionnary = (generate_a_dictionnary(prior_value_a,prior_value_a))
Ninstances = 10
Ntrials = 500
overwrite = False
run_models(savepath,models_dictionnary,Ntrials,Ninstances,overwrite=overwrite,verbose = True)

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