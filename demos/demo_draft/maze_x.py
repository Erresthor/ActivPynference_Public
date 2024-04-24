import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import sys,os,inspect
import math

from actynf.base.miscellaneous_toolbox import isField
from actynf.base.function_toolbox import normalize

from actynf.layer.model_layer import mdp_layer
from actynf.layer.layer_link import establish_layerLink
from actynf.architecture.network import network

def lerp(a, b, t):
        return a*(1 - t) + b*t

# Python version of matlab functions :
def sub2ind(array_shape, sub_tuple):
    rows, cols = sub_tuple[0],sub_tuple[1]
    if ((rows < 0)or(rows>=array_shape[0])) or ((cols < 0)or(cols>=array_shape[1])) :
        raise ValueError(str(sub_tuple) + " is outside the range for array shape " + str(array_shape))
    return cols*array_shape[0] + rows

def ind2sub(array_shape, ind):
    rows = (ind // array_shape[1])
    cols = (ind % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return rows, cols

def build_maze(
        maze_array,start_idx,end_idx,
        T,Th,seed=None):
    # maze_array = np.swapaxes(maze_array,0,1).T

    # maze_array[start_idx] = 5
    # maze_array[end_idx] = 6
    flattened_maze = maze_array.flatten('F')
    
    Ns = flattened_maze.shape[0]
    start_pos = sub2ind(maze_array.shape,start_idx)
    end_pos = sub2ind(maze_array.shape,end_idx)

    print("GOAL : " + str(end_pos))
    print("START : " + str(start_pos))
    # print(start_pos)
    # print(Ns)
    d = [np.zeros((Ns,))]
    d[0][start_pos] = 1

    a0 = np.zeros((2,Ns))  # What is on that maze tile ?
    a0[0,:] = np.ones((Ns,)) - flattened_maze
    a0[1,:] = flattened_maze

    a1 = np.eye(Ns)        # Where on the maze am I ?

    a = [a0, a1]

    u   = np.array([[ 1, 0],
                    [-1, 0],
                    [ 0, 1],
                    [ 0,-1],
                    [ 0, 0]]) # allowable actions
    Nu = u.shape[0]

    B = np.zeros((Ns,Ns,Nu))
    for from_x in range(maze_array.shape[0]):
        for from_y in range(maze_array.shape[1]):
            s = sub2ind(maze_array.shape,(from_x,from_y))
            for u_ix in range(Nu):
                try :
                    ss = sub2ind(maze_array.shape,(from_x + u[u_ix,0] ,from_y + + u[u_ix,1]))
                    B[ss,s,u_ix] = 1
                except:
                    B[s,s,u_ix] = 1
    b = [B]

    c1 = np.array([2,-2])
    c2 = np.zeros((Ns,))
    Xtarget,Ytarget = end_idx[0],end_idx[1]
    Xtarget,Ytarget = end_idx[1],end_idx[0]
    for c_ix in range(Ns):
        x,y = ind2sub(maze_array.shape,c_ix)
        c2[c_ix] = -1.0*np.sqrt((Xtarget-x)*(Xtarget-x)+(Ytarget-y)*(Ytarget-y))# - 1.0
    #c2[sub2ind(maze_array.shape,(Ytarget,Xtarget))] = 0
    c = [c1,c2]

    U = np.array(range(Nu))
    e = np.ones(U.shape)
    maze_process = mdp_layer("maze_environment","process",a,b,c,d,e,U,T,Th,in_seed=seed)
    return maze_process

def maze_model(maze_process,seed=None,
               initial_tile_confidence=1.0):
    T = maze_process.T
    Th = maze_process.T_horizon

    # a = [2*np.ones(maze_process.a[0].shape),255*maze_process.a[1]]
    # a = maze_process.a


    # observation_a = 10*maze_process.a[1] + np.ones(maze_process.a[1].shape)
    observation_a = 200*maze_process.a[1]

    a = [initial_tile_confidence*np.ones(maze_process.a[0].shape),observation_a]
    b = maze_process.b
    c = maze_process.c
    d = [np.ones(maze_process.d[0].shape)]
    e = maze_process.e
    U = maze_process.U
    maze_model = mdp_layer("maze_model","model",
                    a,b,c,d,e,U,T,Th,in_seed=seed)
    
    maze_model.learn_options.learn_a = True
    maze_model.learn_options.learn_b = False
    maze_model.learn_options.learn_d = True
    maze_model.hyperparams.alpha = 10
    return maze_model

def get_maze_network(maze_array,start_idx,end_idx,
                     T,Th,seeds,
                     init_conf=1.0):
    proc = build_maze(maze_array,start_idx,end_idx,
                                 T,Th,seeds[0])
    model = maze_model(proc,seeds[1],initial_tile_confidence=init_conf)

    linker_obs = establish_layerLink(proc, model,[["o","o"]])
    linker_act = establish_layerLink(model, proc,[["u","u"]])

    maze_net = network([proc,model],"maze")
    return maze_net

def mazeplot(maze,start_position,target_position,
             t,
             x_d=None,o_d=None,u_d=None,
             a=None,b=None,c=None,d=None,e=None):
    # print(start_position,target_position)
    maze = maze.T
    maze_shape = maze.shape
    sy = 640
    by = (int)(sy/maze.shape[0])
    sx = int(sy*(float(maze_shape[1])/float(maze_shape[0])))
    bx = (int)(sx/maze.shape[1])

    cold_color = np.array([0.1, 0.1, 1,1])
    hot_color = np.array([1, 0.1, 0.1,1])
    maze_img = Image.new("RGB",(sx,sy),"white")
    draw = ImageDraw.Draw(maze_img,"RGBA")


    # if (belief):
    #     pass
    # else :
    #     # DRAW PERCEPTION OF THE MAZE
    #     perception_matrix = normalize(a[0],0)
    #     for k in range((perception_matrix.shape[1])) :
    #         values = perception_matrix[:,k]
    #         indices = ind2sub(maze.shape,k)
    #         position = [bx/2.0 + indices[0]*bx*1.0 ,by/2.0 + indices[1]*by*1.0]
    #         rectangle_up = (position[0]-bx/2.0,position[1]-by/2.0)
    #         rectangle_down = (position[0]+bx/2.0,position[1]+by/2.0)
    #         color = np.array([1, 1, 1])*values[0]
    #         color = tuple((color*255).astype(int))
    #         draw.rectangle((rectangle_up,rectangle_down),fill=color)

    if (isField(a)):
        # DRAW MAZE CELLS
        maze_arr = normalize(a[0],0)
    else :
        true_maze = maze.flatten('F')
        maze_arr = np.zeros((2,true_maze.shape[0]))
        maze_arr[0,:] = np.ones((true_maze.shape[0],)) - true_maze
        maze_arr[1,:] = true_maze
    
    for k in range((maze_arr.shape[1])) :
        values = maze_arr[:,k]
        indices = ind2sub(maze.shape,k)
        # position = [ by/2.0 + indices[1]*by*1.0,bx/2.0 + indices[0]*bx*1.0]
        position = [ by/2.0 + indices[0]*by*1.0,bx/2.0 + indices[1]*bx*1.0]

        color = np.array([0, 0, 0, 1-values[0]])
        color = tuple((color*255.0).astype(int))

        draw.line(((position[0]-bx/2.0,position[1]-by/2.0),(position[0]+bx/2.0,position[1]+by/2.0)),width = 10,fill=color)
        draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color)

        draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]+by/2.0)),width = 10,fill=color)
        draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]-bx/2.0,position[1]-by/2.0)),width = 10,fill=color)
        draw.line(((position[0]+bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color)
        draw.line(((position[0]-bx/2.0,position[1]-by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color)


    # DRAW SUBJECT PREFERENCES
    if (isField(c)):
        pref_matrix = c[1]
        min_pref = np.min(pref_matrix)
        for k in range((pref_matrix.shape[0])) :
            values = pref_matrix[k]
            indices = ind2sub(maze.shape,k)
            # position = [ by/2.0 + indices[1]*by*1.0,bx/2.0 + indices[0]*bx*1.0]
            position = [ by/2.0 + indices[0]*by*1.0,bx/2.0 + indices[1]*bx*1.0]
            color = tuple((lerp(cold_color,hot_color,1-(values/min_pref))*255).astype(int))
            # print(values,color)
            factor = 0.5

            rectangle_up = (position[0]-factor*bx/2.0,position[1]-factor*by/2.0)
            rectangle_down = (position[0]+factor*bx/2.0,position[1]+factor*by/2.0)
            draw.rectangle((rectangle_up,rectangle_down),fill=color)

    # else :
    #     # DRAW ACTUAL MAZE
    #     for y in range(maze_shape[0]):
    #         for x in range (maze_shape[1]):
    #             position = [bx/2.0 + x*bx*1.0 ,by/2.0 + y*by*1.0]

    #             if (maze[x,y]>0.5) :
    #                 color = "black"
    #                 draw.line(((position[0]-bx/2.0,position[1]-by/2.0),(position[0]+bx/2.0,position[1]+by/2.0)),width = 10,fill=color)
    #                 draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color)

    #                 draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]+by/2.0)),width = 10,fill=color)
    #                 draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]-bx/2.0,position[1]-by/2.0)),width = 10,fill=color)
    #                 draw.line(((position[0]+bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color)
    #                 draw.line(((position[0]-bx/2.0,position[1]-by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color)

    # DRAW TARGET
    color = "red"
    position = [bx/2.0 + target_position[0]*bx*1.0 ,by/2.0 + target_position[1]*by*1.0]
    position = [bx/2.0 + target_position[1]*bx*1.0 ,by/2.0 + target_position[0]*by*1.0]
    draw.line(((position[0]-bx/2.0,position[1]-by/2.0),(position[0]+bx/2.0,position[1]+by/2.0)),width = 10,fill=color)
    draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color)

    # DRAW BEGINNING
    color = "green"
    position = [bx/2.0 + start_position[0]*bx*1.0 ,by/2.0 + start_position[1]*by*1.0]
    position = [bx/2.0 + start_position[1]*bx*1.0 ,by/2.0 + start_position[0]*by*1.0]
    draw.line(((position[0]-bx/2.0,position[1]-by/2.0),(position[0]+bx/2.0,position[1]+by/2.0)),width = 10,fill=color)
    draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color)

    # # DRAW CURRENT
    # current_position = list(ind2sub(maze.shape,current_position))
    # color = "purple"
    # shift = 0.1*bx
    # position = [bx/2.0 + current_position[1]*bx*1.0 ,by/2.0 + current_position[0]*by*1.0 + shift]
    # draw.line(((position[0]-bx/2.0 + shift ,position[1]-by/2.0),(position[0]+bx/2.0 - shift,position[1]-by/2.0)),width = 10,fill=color)
    # draw.line(((position[0],position[1]),(position[0]+bx/2.0 - shift,position[1]-by/2.0)),width = 10,fill=color)
    # draw.line(((position[0],position[1]),(position[0]-bx/2.0 + shift,position[1]-by/2.0)),width = 10,fill=color)

    # DRAW LAYER STATES DISTRIBUTION 
    # - belief if model, true dist if process
    if (isField(x_d)):
        for state in range(x_d.shape[0]):
            current_state_coord = ind2sub(maze.shape,state)
            # position = [bx/2.0 + current_state_coord[1]*bx*1.0 ,by/2.0 + current_state_coord[0]*by*1.0]
            position = [bx/2.0 + current_state_coord[0]*bx*1.0 ,by/2.0 + current_state_coord[1]*by*1.0]

            factor  = x_d[state,t]*0.9
            rectangle_up = (position[0]-factor*bx/2.0,position[1]-factor*by/2.0)
            rectangle_down = (position[0]+factor*bx/2.0,position[1]+factor*by/2.0)

            color = (lerp(cold_color,hot_color,x_d[state,t]))
            color = tuple((color*255).astype(int))
            
            color="blue"
            draw.rectangle((rectangle_up,rectangle_down),fill=color,outline="black")

    #maze_img.save('outal.png',"PNG")
    return maze_img
            

def mazelayerplot(maze,start_position,target_position,
                t,layer):
    x_d = layer.STM.x_d
    o_d = layer.STM.o_d
    u_d = layer.STM.u_d,
    a = layer.a
    b = layer.b
    c = layer.c
    d = layer.d
    e = layer.e 
    return mazeplot(maze,start_position,target_position,
             t,
             x_d,o_d,u_d,
             a,b,c,d,e)

def run_maze_example(Ntrials,initial_a_confidence,save_gif_to_path):
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
    my_network = get_maze_network(maze_array,start_idx,end_idx,
                                  T,Th,[1,2],initial_a_confidence)
    print(my_network)

    


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
    Ntrials = 20
    initial_a_confidence = 1.0
    result_gif_path = os.path.join(os.path.dirname(__file__),"plots","maxe_x_plot__" + str(initial_a_confidence) + "__.gif")
    run_maze_example(Ntrials,initial_a_confidence,result_gif_path)