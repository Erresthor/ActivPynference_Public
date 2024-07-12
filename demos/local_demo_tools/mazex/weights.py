import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import sys,os,inspect
import math

from scipy.interpolate import splprep,splev

from actynf.base.miscellaneous_toolbox import isField
from actynf.base.function_toolbox import normalize

from actynf import layer
from actynf import link
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

def build_maze(maze_array,start_idx,end_idx,dirac_goal=False,p_transition=1.0):
    flattened_maze = maze_array.flatten('F')
    
    Ns = flattened_maze.shape[0]
    start_pos = sub2ind(maze_array.shape,start_idx)
    end_pos = sub2ind(maze_array.shape,end_idx)
    # print("GOAL : " + str(end_pos))
    # print("START : " + str(start_pos))

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
                    ss = sub2ind(maze_array.shape,(from_x + u[u_ix,0] ,from_y + u[u_ix,1]))
                    B[ss,s,u_ix] = p_transition
                    B[s,s,u_ix] = 1 - p_transition
                except:
                    B[s,s,u_ix] = 1
    b = [B]

    c1 = np.array([2,-2])
    c2 = np.zeros((Ns,))
    # Xtarget,Ytarget = end_idx[0],end_idx[1]
    Xtarget,Ytarget = end_idx[1],end_idx[0]
    for c_ix in range(Ns):
        x,y = ind2sub(maze_array.shape,c_ix)
        if dirac_goal:
            c2[c_ix] = -1.0
        else :
            c2[c_ix] = -1.0*np.sqrt((Xtarget-x)*(Xtarget-x)+(Ytarget-y)*(Ytarget-y))# - 1.0
    if dirac_goal:
        c2[sub2ind(maze_array.shape,(Ytarget,Xtarget))] = 100
    c = [c1,c2]

    U = np.expand_dims(np.array(range(Nu)),-1)
    e = np.ones(U.shape)
    return a,b,c,d,e,U


# Actynf layer building
def get_maze_process_layer(maze_array,start_idx,end_idx,
                            T,Th,dirac_goal=False,seed=None):
    a,b,c,d,e,U = build_maze(maze_array,start_idx,end_idx,dirac_goal=dirac_goal)
    maze_process = layer("maze_environment","process",a,b,c,d,e,U,T,Th,in_seed=seed)
    return maze_process

def build_maze_model(maze_array,start_idx,end_idx,
                     initial_tile_confidence=1.0,
                     rs=1.0,la=-2,
                     dirac_goal=False,
                     p_transition=1.0):
    flattened_maze = maze_array.flatten('F')
    
    Ns = flattened_maze.shape[0]
    start_pos = sub2ind(maze_array.shape,start_idx)
    end_pos = sub2ind(maze_array.shape,end_idx)
    
    content_observation = initial_tile_confidence*np.ones((2,Ns))
        # What's on this tile ? (prefered vs adversive stimulus)
    map_observation = 200*np.eye(Ns) + 1.0*np.ones((Ns,Ns))
        # Where are we on the map ?
    a = [content_observation,map_observation]
    
    # Let's assume the transitions are known !
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
                    ss = sub2ind(maze_array.shape,(from_x + u[u_ix,0] ,from_y + u[u_ix,1]))
                    B[ss,s,u_ix] = p_transition
                    B[s,s,u_ix] = 1 - p_transition
                except:
                    B[s,s,u_ix] = 1
    b = [B]
    
    Xtarget,Ytarget = end_idx[1],end_idx[0]
    c1 = np.array([2,la])
    c2 = np.zeros((Ns,))
    for c_ix in range(Ns):
        x,y = ind2sub(maze_array.shape,c_ix)
        if dirac_goal:
            c2[c_ix] = -1.0*rs
        else :
            c2[c_ix] = -1.0*rs*np.sqrt((Xtarget-x)*(Xtarget-x)+(Ytarget-y)*(Ytarget-y))# - 1.0
    if dirac_goal:
        c2[sub2ind(maze_array.shape,(Ytarget,Xtarget))] = rs
    c = [c1,c2]
    
    d = [np.ones((Ns,))]
    # d[0][start_pos] = 1
    
    U = np.expand_dims(np.array(range(Nu)),-1)
    e = np.ones(U.shape)
    return a,b,c,d,e,U

def get_maze_model_layer(maze_array,start_idx,end_idx,
                         T,Th,initial_tile_confidence=1.0,rs=1.0,la=-2,
                         seed=None,dirac_goal=False,alpha=16,p_transition=1.0
                         ):
    
    a,b,c,d,e,U = build_maze_model(maze_array,start_idx,end_idx,
                            initial_tile_confidence=initial_tile_confidence,
                            rs = rs,la = la,
                            dirac_goal=dirac_goal,p_transition=p_transition)
    
    maze_model = layer("maze_model","model",
                    a,b,c,d,e,U,T,Th,in_seed=seed)
    
    maze_model.learn_options.learn_a = True
    maze_model.learn_options.learn_b = False
    maze_model.learn_options.learn_d = True
    maze_model.hyperparams.alpha = alpha
    return maze_model

def get_maze_network(maze_array,start_idx,end_idx,
                     T,Th,seeds,
                     init_conf=1.0,rs=1.0,la=2,
                     dirac_goal=False,alpha=16,
                     seek_a_novelty = False,p_transition=1.0):
    proc = get_maze_process_layer(maze_array,start_idx,end_idx,
                        T,Th,dirac_goal=dirac_goal,seed=seeds[0],p_transition=p_transition)
    model = get_maze_model_layer(maze_array,start_idx,end_idx,T,Th,
                        initial_tile_confidence=init_conf,rs=rs,la=la,
                        seed=seeds[1],dirac_goal=dirac_goal,alpha=alpha,p_transition=p_transition)
    model.hyperparams.a_novelty = seek_a_novelty
    

    proc.inputs.u = link(model,lambda x : x.u)
    model.inputs.o = link(proc,lambda x : x.o)

    maze_net = network([proc,model],"maze")
    return maze_net

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
    