import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import sys,os,inspect
import math
from scipy.interpolate import splprep,splev

from actynf.base.miscellaneous_toolbox import isField
from actynf.base.function_toolbox import normalize

from .weights import ind2sub,lerp,sub2ind
from .weights import get_maze_network

# Plotting functions
def maze_bckg(maze,start_position,target_position,
             a=None,show_start_and_stop= True,
             image_size=640):   

    # maze = maze.T
    maze_shape = maze.shape
    
    # Compute image size : 
    sy = image_size
    by = (int)(sy/maze.shape[0])
    sx = int(sy*(float(maze_shape[1])/float(maze_shape[0])))
    bx = (int)(sx/maze.shape[1])

    maze_img = Image.new("RGB",(sx,sy),"white")
    draw = ImageDraw.Draw(maze_img,"RGBA")

    if (isField(a)):
        # DRAW MAZE CELLS
        maze_arr = normalize(a[0],0)
    else :
        true_maze = maze.flatten('F')
        maze_arr = np.zeros((2,true_maze.shape[0]))
        maze_arr[0,:] = np.ones((true_maze.shape[0],)) - true_maze
        maze_arr[1,:] = true_maze
    
    # Draw cells
    for k in range((maze_arr.shape[1])) :
        values = maze_arr[:,k]
        init_color_fill = 0.2
        color_fill = np.array([init_color_fill, init_color_fill, init_color_fill, 1-values[0]])
        color_fill = tuple((color_fill*255.0).astype(int))

        init_color_stroke = 0.0
        color_stroke = np.array([init_color_stroke, init_color_stroke, init_color_stroke, 1-values[0]])
        color_stroke = tuple((color_stroke*255.0).astype(int))

        indices = ind2sub(maze.shape,k)
        position = [ by/2.0 + indices[0]*by*1.0,bx/2.0 + indices[1]*bx*1.0]

        draw.rectangle([(position[0]-bx/2.0, position[1]-by/2.0), (position[0]+bx/2.0,position[1]+by/2.0)],fill=color_fill)
        draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]+by/2.0)),width = 10,fill=color_stroke)
        draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]-bx/2.0,position[1]-by/2.0)),width = 10,fill=color_stroke)
        draw.line(((position[0]+bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color_stroke)
        draw.line(((position[0]-bx/2.0,position[1]-by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color_stroke)
    
    if show_start_and_stop:
        ratio = 0.7 # for drawing start and stop
        
        # DRAW TARGET
        color = "red"
        position = [bx/2.0 + target_position[1]*bx*1.0 ,by/2.0 + target_position[0]*by*1.0]
        draw.line(((position[0]-ratio*bx/2.0,position[1]-ratio*by/2.0),(position[0]+ratio*bx/2.0,position[1]+ratio*by/2.0)),width = 10,fill=color)
        draw.line(((position[0]-ratio*bx/2.0,position[1]+ratio*by/2.0),(position[0]+ratio*bx/2.0,position[1]-ratio*by/2.0)),width = 10,fill=color)

        # DRAW BEGINNING
        color = "green"
        position = [bx/2.0 + start_position[1]*bx*1.0 ,by/2.0 + start_position[0]*by*1.0]
        draw.line(((position[0]-ratio*bx/2.0,position[1]-ratio*by/2.0),(position[0]+ratio*bx/2.0,position[1]+ratio*by/2.0)),width = 10,fill=color)
        draw.line(((position[0]-ratio*bx/2.0,position[1]+ratio*by/2.0),(position[0]+ratio*bx/2.0,position[1]-ratio*by/2.0)),width = 10,fill=color)

    return maze_img,draw
   
def mazeplot(maze,start_position,target_position,
             t=0,
             x_d=None,o_d=None,u_d=None,
             a=None,b=None,c=None,d=None,e=None,
             show_start_and_stop= True,
             image_size=640):
    """ Plot the background and some other elements on top !"""
    
    
    # print(start_position,target_position)
    maze = maze
    maze_shape = maze.shape
    sy = image_size
    by = (int)(sy/maze.shape[0])
    sx = int(sy*(float(maze_shape[1])/float(maze_shape[0])))
    bx = (int)(sx/maze.shape[1])

    cold_color = np.array([0.1, 0.1, 1,0.5])
    hot_color = np.array([1, 0.1, 0.1,0.5])

    maze_img,draw = maze_bckg(maze,start_position,target_position,
             a=a,show_start_and_stop= show_start_and_stop,
             image_size=image_size)
    
    
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
            draw.ellipse((rectangle_up,rectangle_down),fill=color)#draw.rectangle((rectangle_up,rectangle_down),fill=color)

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
    u_d = layer.STM.u_d
    a = layer.a
    b = layer.b
    c = layer.c
    d = layer.d
    e = layer.e 
    return mazeplot(maze,start_position,target_position,
             t,
             x_d,o_d,u_d,
             a,b,c,d,e)


# Examples
# 1. Generate a gif of the agent moving through the maze
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

def gif_plots(save_to_file=None):
    if save_to_file == None :
        save_to_file = os.path.dirname(__file__)
    
    # To get the gifs:
    Ntrials = 20
    initial_a_confidence = 1.0
    
    result_gif_path = os.path.join(save_to_file,"plots","maxe_x_plot__" + str(initial_a_confidence) + "__.gif")
    run_maze_example(Ntrials,initial_a_confidence,result_gif_path)


# 2. Plot the exploration trajectories of an agent
def plot_trajectory(ax,maze_array,state_hist,image_size=640,lw=1,col=np.array([0.5,0.6,1.0,1.0])):   
    ratio = 0.1
    
    position_prev = np.array([-1e10,-1e10])
    L = []
    for state in state_hist:
        state_coord = np.array(ind2sub(maze_array.shape,state))
        
        sy = image_size
        by = (int)(sy/maze_array.shape[0])
        sx = int(sy*(float(maze_array.shape[1])/float(maze_array.shape[0])))
        bx = (int)(sx/maze_array.shape[1])
        
        position_raw = np.array([ by/2.0 + state_coord[0]*by*1.0,bx/2.0 + state_coord[1]*bx*1.0])
        if np.linalg.norm(position_raw - position_prev) < 1e-5: # If we did not move, place no point
            pass
        else :
            position_prev = position_raw
            
            
            random_minusratio_ratio = 2.0*(np.random.random(2,)-0.5)*ratio
            scaled_noise = random_minusratio_ratio*np.array([bx,by])
            
            position = position_raw + scaled_noise
            
            L.append(position)
    Larr = np.array(L)
    x, y = zip(*Larr)
    # print(x,y)
    f, u = splprep([x, y], s=0)#, per=True)
    
    xint, yint = splev(np.linspace(0, 1, 100), f)
    
    ax.scatter(x, y,c=np.array([col]))
    ax.plot(xint, yint,color=col,linewidth=lw)

def plot_grid_and_trajectory(ax,
            maze_array,start_idx,end_idx,
            a,trajectory,
            image_size = 640):
    """Plots the last trial stored in this network STM !

    Args:
        maze_array (_type_): _description_
        start_idx (_type_): _description_
        end_idx (_type_): _description_
        my_network (_type_): _description_
        image_size (int, optional): _description_. Defaults to 640.

    Returns:
        _type_: _description_
    """
    # PLOT A FIGURE FOR EACH TRIAL WITH LEARNING AND TRAJ
    # print(my_network.layers[0].STM.x)
    # fig,ax = plt.subplots()
    
    maze_img,draw = maze_bckg(maze_array,start_idx,end_idx,
            a=a,show_start_and_stop= True,
            image_size=image_size)
    ax.imshow(maze_img)
    
    plot_trajectory(ax,maze_array,trajectory,image_size)
    
    ax.set_axis_off()
    return ax

def plot_grid_and_trajectories(ax,
                maze_array,start_idx,end_idx,
                a,trajectory_list,
                image_size = 640,
                from_color=np.array([0.0,0.5,1.0,0.4]),
                to_color=  np.array([1.0,0.1,0.0,0.5])):
    
    maze_img,draw = maze_bckg(maze_array,start_idx,end_idx,
            a=a,show_start_and_stop= True,
            image_size=image_size)
    ax.imshow(maze_img)
    
    Ntrials = len(trajectory_list)
    for trial in range(Ntrials):
        
        color_trajectory = lerp(from_color,to_color,(trial/(Ntrials-1)))
        # print((trial/Ntrials-1))
        plot_trajectory(ax,maze_array,trajectory_list[trial],image_size,lw=3,col = color_trajectory)
    
    ax.set_axis_off()
    return ax


# Plot full figures
def several_trajectory_learning_plots(maze_array,start_idx,end_idx,
            Ntrials = 20,T = 14,Th = 5,
            initial_a_confidence = 0.1,
            rs = 2.0,
            la = 2.0,
            dirac_goal=False,
            alpha=16,
            image_size = 640,
            seeds = [240,2]):

    my_network = get_maze_network(maze_array,start_idx,end_idx,
                     T,Th,seeds,
                     init_conf=initial_a_confidence,rs=rs,la=la,
                     dirac_goal=dirac_goal,alpha=alpha)
    
    for trial in range(Ntrials):

        my_network.run()

        a = my_network.layers[1].a
        state_trajectory = my_network.layers[0].STM.x[0,:]

        fig,trial_axes = plt.subplot()
        
        plot_grid_and_trajectory(trial_axes,
            maze_array,start_idx,end_idx,
            a,state_trajectory,
            image_size = image_size)
        
        fig.suptitle("Trial " + str(trial))
        fig.show()

def all_trajectories_plots(maze_array,start_idx,end_idx,
            Ntrials = 20,T = 14,Th = 5,
            initial_a_confidence = 0.1,
            rs = 2.0,
            la = 2.0,
            dirac_goal=False,
            alpha=16,
            image_size = 640,
            seeds = [240,2]):

    my_network = get_maze_network(maze_array,start_idx,end_idx,
                     T,Th,seeds,
                     init_conf=initial_a_confidence,rs=rs,la=la,
                     dirac_goal=dirac_goal,alpha=alpha)
    
    
    list_of_trajectories = []
    for trial in range(Ntrials):

        my_network.run()

        a = my_network.layers[1].a
        state_trajectory = my_network.layers[0].STM.x[0,:]
        
        list_of_trajectories.append(state_trajectory)
        

    fig,trial_axes = plt.subplots()
    
    plot_grid_and_trajectories(trial_axes,
        maze_array,start_idx,end_idx,
        a,list_of_trajectories,
        image_size = image_size)
    
    fig.suptitle("First "+ str(Ntrials) +" trials")
    return fig 
    fig.show()


if __name__ == '__main__':
    # initial_a_confidence = [0.01,0.05,0.1,0.5,1,10,100]
    # for inita in initial_a_confidence:
    #     plots_mazex(inita)
    Ntrials = 20
    T = 14
    Th = 4
    start_idx = (7,1)
    end_idx = (4,4)
    end_idx = (1,6)
    maze_array = np.array([
        [1,1,1,1,1,1,1,1],
        [1,0,0,0,0,1,0,1],
        [1,1,1,0,1,1,0,1],
        [1,1,0,0,0,1,0,1],
        [1,1,0,1,0,0,0,1],
        [1,1,0,1,1,1,0,1],
        [1,0,0,0,0,0,0,1],
        [1,0,1,1,1,1,1,1]
    ])
    my_network = get_maze_network(maze_array,start_idx,end_idx,
                                  T,Th,[1,2],0)

    interp_list = []
    pointlist = []
    for k in range(Ntrials):
        # matrices used for this trial's computations:
        layer = my_network.layers[1]
        a = layer.a
        b = layer.b
        c = layer.c
        d = layer.d
        e = layer.e

        my_network.run()


        # PLOT A FIGURE FOR EACH TRIAL WITH LEARNING AND TRAJ
        print(my_network.layers[0].STM.x)
        L = []
        for state in my_network.layers[0].STM.x[0,:]:
            state_coord = np.array(ind2sub(maze_array.shape,state))
            
            sy = 640
            by = (int)(sy/maze_array.shape[0])
            sx = int(sy*(float(maze_array.shape[1])/float(maze_array.shape[0])))
            bx = (int)(sx/maze_array.shape[1])
            
            random_minus1_1 = 2.0*(np.random.random(2,)-0.5)
            ratio = 0.1
            pos_noise_x,pos_noise_y = random_minus1_1[0]*ratio*bx,random_minus1_1[1]*ratio*by

            position = [ by/2.0 + state_coord[0]*by*1.0 + pos_noise_x,bx/2.0 + state_coord[1]*bx*1.0 + pos_noise_y]
            if (len(L)>0):
                if (position == L[-1]):
                    continue
            L.append(position)
        Larr = np.array(L)
        x, y = zip(*Larr)
        f, u = splprep([x, y], s=0)#, per=True)
        interp_list.append(f)
        pointlist.append([x,y])

    layer_proc = my_network.layers[0]
    maze_img = maze_bckg(maze_array,start_idx,end_idx,
                0,
                layer.STM.x_d,layer.STM.o_d,layer.STM.u_d,
                layer.a,None,None,None,None) 
    # maze_img.show()
    implot = plt.imshow(maze_img)

    hotcol = np.array([255,10,10,100])/255.0
    coldcol = np.array([10,10,255,100])/255.0
    for k,f in enumerate(interp_list):
        colarr = coldcol + (k/(len(interp_list)-1))*(hotcol-coldcol)
        # col = colarr.astype(int)
        col = colarr
        col_allpoints = np.array([colarr])
        xint, yint = splev(np.linspace(0, 1, 100), f)
        # print(col)
        [x,y] = pointlist[k]
        plt.scatter(x, y,c=col_allpoints)
        plt.plot(xint, yint,color=col,linewidth=3)
    # plt.show()
    plt.axis('off')
    plt.savefig(str(0)+".png",bbox_inches='tight')
    plt.close()
    
    
    
    # # initial_a_confidence = [0.01,0.05,0.1,0.5,1,10,100]
    # # for inita in initial_a_confidence:
    # #     plots_mazex(inita)
    # Ntrials = 20
    # T = 14
    # Th = 4
    # start_idx = (7,1)
    # end_idx = (4,4)
    # end_idx = (1,6)
    # maze_array = np.array([
    #     [1,1,1,1,1,1,1,1],
    #     [1,0,0,0,0,1,0,1],
    #     [1,1,1,0,1,1,0,1],
    #     [1,1,0,0,0,1,0,1],
    #     [1,1,0,1,0,0,0,1],
    #     [1,1,0,1,1,1,0,1],
    #     [1,0,0,0,0,0,0,1],
    #     [1,0,1,1,1,1,1,1]
    # ])
    
    
    # plots_mazex(maze_array,start_idx,end_idx,
    #         Ntrials = 20,T = 14,Th = 6,
    #         initial_a_confidence = 0.2,
    #         rs = 0.5,
    #         la = 2.0,
    #         dirac_goal=False,
    #         alpha=3,
    #         image_size = 640,
    #         seeds = [240,2])