#!/usr/bin/python
from json import load
import sys,inspect,os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from pyai.base.file_toolbox import save_flexible,load_flexible
from pyai.layer.layer_learn import MemoryDecayType
from pyai.model.active_model import ActiveModel
from pyai.neurofeedback_run import evaluate_model_mean

from pyai.models_neurofeedback.climb_stairs import nf_model,evaluate_container

import time as t

# We grab a "performance_ouptut.pyai" file and extract the data :D
def load_perf(filepath):
    return (load_flexible(filepath))

def generate_grids(model_list,pick_t,parameter_index,size = (12,12),smooth_it = 0):
    Xgrid = np.zeros(size)
    Ygrid = np.zeros(size)
    Zgrid = np.zeros(size)
    for model in model_list :
        model_object = model[0]
        results_list = model[1]
        options = model_object.input_parameters
        index = tuple(model_object.index)
        #print(index)
        #print(options[1],options[4])
        Xgrid[index] = options[1]
        Ygrid[index] = options[4]
        Zgrid[index] = results_list[parameter_index][pick_t]
        if(smooth_it > 0):
            sum = 0
            cnt = 0
            for k in range(pick_t-smooth_it,pick_t + smooth_it + 1):
                try :
                    sum += results_list[parameter_index][k]
                    cnt += 1
                except :
                    sum += 0
                    cnt += 0
            Zgrid[index] = sum/cnt
    return Xgrid,Ygrid,Zgrid


if __name__=="__main__":
    savepath = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","code","results","article_1")
    filename = "simulation_output_002.pyai"
    t0 = t.time()
    big_list = load_perf(os.path.join(savepath,filename))
    timefloat = (t.time()-t0)
    format_float = "{:.2f}".format(timefloat)
    print("Loaded performance file in " + format_float + " seconds.")
    

    size = (21,21)
    param_plot = 3
    
    N = 50
    fps = 30
    frn = 250

    z_array = np.zeros(size+(frn,))
    for t in range(frn):
        smooth_it = 0
        if (param_plot==8) or (param_plot==9) or (param_plot==10) or (param_plot==11):
            smooth_it = 15
        x,y,z = generate_grids(big_list,t,param_plot,size=size,smooth_it=smooth_it)
        z_array[:,:,t] = z
    
    # limit = 999
    # x = x[:limit,:limit]
    # y = y[:limit,:limit]
    # z_array = z_array[:limit,:limit]


    def change_plot(frame_number, zarray, plot):
        plot[0].remove()
        #plot[0] = ax.plot_surface(x, y, zarray[:, :, frame_number], cmap="afmhot_r")
        plot[0] = ax.plot_surface(x, y, zarray[:, :, frame_number], cmap=cm.coolwarm,linewidth=0, antialiased=False)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('a accuracy')
    ax.set_ylabel('b accuracy')



    if (param_plot==3):
        ax.set_zlabel('PERCEPTION MODEL error')
        ax.set_title('PERCEPTION MODEL error - KL(a,A)')
    elif  (param_plot==4):
        ax.set_zlabel('ACTION MODEL error')
        ax.set_title('ACTION MODEL error - KL(b,B)')
    elif  (param_plot==5):
        ax.set_title('PERCEPTION MODEL entropy - E(a)')
        ax.set_zlabel('PERCEPTION MODEL entropy')
    elif  (param_plot==6):
        ax.set_zlabel('ACTION MODEL entropy')
        ax.set_title('ACTION MODEL entropy - E(b)')
    elif  (param_plot==7):
        ax.set_zlabel('INITIAL STATE MODEL entropy')
        ax.set_title('INITIAL STATE MODEL entropy - E(d)')
    elif  (param_plot==8):
        ax.set_zlabel('state error')
        ax.set_title('TRUE STATE error - ||optimal,true||_s')
    elif  (param_plot==9):
        ax.set_zlabel('actions taken error')
        ax.set_title('TRUE ACTIONS TAKEN error - ||optimal,true||_u')
    elif  (param_plot==10):
        ax.set_zlabel('observations error')
        ax.set_title('TRUE OBSERVATIONS error - ||optimal,true||_o')
    elif  (param_plot==11):
        ax.set_zlabel('perception error')
        ax.set_title('TRUE PERCEPTION error - KL(s_estimated,s_true)')

    plot = [ax.plot_surface(x, y, z_array[:, :, 0], color='0.75', rstride=1, cstride=1)]
    ax.set_zlim(0, 1.1)
    ani = animation.FuncAnimation(fig, change_plot, frn, fargs=(z_array, plot), interval=1000 / fps)
    plt.show()
    
    
    # t = np.arange(0,500,1)
    # print(len(big_list))
    # pick_t = 400
    

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                    linewidth=0, antialiased=False)


    # ax.axes.set_xlim3d(left=1, right=5)
    # ax.axes.set_ylim3d(bottom=1, top=5)

    
    # We have a problem : these simulations are not ordered by design (we stack them up in a list :( )
    # Solution 1 : stack them up in an other form of data structure to preserve spatial coherence
    # SOlution 2 : sort them now depending on their variables
    # Solution 3 : include an "index" variable in ActiveModel class. (but we'd have to implement updates
    # of the model object between sessions.)




















