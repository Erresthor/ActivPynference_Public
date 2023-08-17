import sys,os
import random
import numpy as np
import matplotlib.pyplot as plt


from demo.new_layer_test import run_test
from demo.link_test import linktest,network_test
from t_maze_test import example_tmaze_plot
from demo.maze_x import build_maze,get_maze_network,mazeplot

from architecture.network import network
from base.function_toolbox import prune_tree_auto


class basic_object:
    def __init__(self):
        self.o_d = np.array([[0.2,0.3,0.3],
                             [0.2,0  ,  0]])
        
        self.x_d = np.array([[0.1,0.2],
                             [0.5,0.1],
                             [0.1,0  ]])

    def __str__(self):
        return_this ="-----------------------\n"
        return_this +='o_d\n'
        return_this +=str(np.round(self.o_d,2))
        return_this +='\nx_d\n'
        return_this +=str(np.round(self.x_d,2))
        return_this +="\n-----------------------\n"
        return return_this



def run_maze_example(save_gif_to_path):
    # Generate the network --------------------------------
    # -----------------------------------------------------
    T = 14
    Th = 4
    start_idx = (7,1)
    end_idx = (4,4)
    end_idx = (1,6)
    maze_array = np.array([
        [1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,1],
        [1,1,1,0,1,1,0,1],
        [1,1,0,0,0,1,0,1],
        [1,1,0,1,0,0,0,1],
        [1,1,0,1,1,1,0,1],
        [1,0,0,0,0,0,0,1],
        [1,0,1,1,1,1,1,1]
    ])
    ic = 1
    my_network = get_maze_network(maze_array,start_idx,end_idx,
                                  T,Th,[1,2],ic)
    print(my_network)

    Ntrials = 50


    # # Just run the network --------------------------------
    # # -----------------------------------------------------
    # inspect = [1000]
    # for k in range(Ntrials):
    #     if (k in inspect):
    #         print("===================================")
    #         print("     " + str(k))
    #         print("===================================")
    #         my_network.layers[1].debug = True
    #     else : 
    #         my_network.layers[1].debug = False
    #     my_network.run()
    #     if (k in inspect):
    #         print(my_network.layers[1].STM.o)
    #         print(my_network.layers[1].STM.u)
    #         print(my_network.layers[1].STM.x_d)
    #         print(my_network.layers[1].a[0])

    # Run the network & generate animation ----------------
    # -----------------------------------------------------
    imglist = []
    for k in range(Ntrials):
        # matrices used for this trial's computations:
        layer = my_network.layers[1]
        a = layer.a
        b = layer.b
        c = layer.c
        d = layer.d
        e = layer.e

        my_network.run()
        for t in range(T):
            maze_img = mazeplot(maze_array,start_idx,end_idx,
                    t,
                    layer.STM.x_d,layer.STM.o_d,layer.STM.u_d,
                    a,None,None,None,None)          
            imglist.append(maze_img)
    imglist[0].save(save_gif_to_path,
               save_all=True, append_images=imglist[1:], optimize=False, duration=100, loop=0)

if __name__ == '__main__':
    result_gif_path = os.path.join("C:",os.sep,"Users","annic","Desktop","Phd","code","active_pynference_local","PyAI","demo","plots","maxe_x_plot__" + str(ic) + "__.gif")
    run_maze_example(result_gif_path)