# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 2021

@author: cjsan
"""
import numpy as np
import random
from PIL import Image, ImageDraw
import sys,os,inspect
import math


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from base.spm12_implementation import MDP 
from base.miscellaneous_toolbox import flexible_copy 
from base.function_toolbox import normalize
from mdp_layer import mdp_layer

def maze_model():
    print("Maze X model --- Model set-up ...  ",end='')
    #Points within a trial
    T = 3
    Ni = 16
    
    MAZE = np.array([[1,0,1,1,1,1,1,1],
                     [1,0,0,0,0,0,0,1],
                     [1,1,1,0,1,1,0,1],
                     [1,1,0,0,0,1,0,1],
                     [1,1,0,1,0,1,0,1],
                     [1,1,0,1,0,1,0,1],
                     [1,0,0,0,0,0,0,1],
                     [1,0,1,1,1,1,1,1]])
    
    END_COORD =  (1,6)
    START_COORD =(7,1)

    START = np.ravel_multi_index(START_COORD, dims=MAZE.shape, order='F')
    END = np.ravel_multi_index(END_COORD, dims=MAZE.shape, order='F')
    flattened_maze = MAZE.flatten(order = 'F')
    # Priors about initial states
    # Prior probabilities about initial states in the generative process
    D_ =[]

    # Context state factor
    D_.append(np.zeros((MAZE.size,)))
    D_[0][flattened_maze<0.5] = 1
    Ns = [D_[0].size]

    A_ = []

    A_what = np.zeros((2,)+tuple(Ns))
    A_what[0,:] = 1 - flattened_maze
    A_what[1,:] = flattened_maze
    A_.append(A_what)
    A_where = np.eye(Ns[0])
    A_.append(A_where)

    a_ = flexible_copy(A_)
    a_[0] = normalize(np.ones((a_[0].shape)),0)
    a_[0][0,:] = 0.75
    a_[0][1,:] = 0.25

    Nmod = len(A_)
    No = []
    for mod in range(Nmod):
        No.append(A_[mod].shape[1])
    
    u = np.array([[1,0],
                [-1,0],
                [0,1],
                [0,-1],
                [0,0]])
    nu = u.shape[0]

    U_ = np.zeros((nu,len(Ns)))
    U_[:,0] = range(nu)
    U_ = U_.astype(np.int)
    B_ = []
    for factor in range(len(Ns)):
        B_ .append(np.zeros((Ns[factor],Ns[factor],nu)))
    for i in range(MAZE.shape[0]):
        for j in range(MAZE.shape[0]):
            # we are in state s:
            s  = np.ravel_multi_index((i,j), dims=MAZE.shape, order='F')
            for k in range(nu):
                try :
                    ss = np.ravel_multi_index((i+u[k,0],j+u[k,1]), dims=MAZE.shape, order='F')
                    # if (flattened_maze[ss]==1):
                    #     B_[0][s,s,k] = 1
                    # else :
                    #     B_[0][ss,s,k] = 1
                    B_[0][ss,s,k] = 1
                except :
                    B_[0][s,s,k] = 1
    

    def sub2ind(array_shape, coords):
        cols = coords[1]
        rows = coords[0]
        return cols*array_shape[0] + rows
    
    def ind2sub(array_shape, ind):
        # for 2 d arrays only
        rows = ind//(array_shape[0])
        cols = ind%(array_shape[0])
        return cols,rows
    
    
    C_ = []
    C_.append(np.array([[2],[-9]]))

    C_target = np.zeros((Ns[0],1))
    K = np.zeros((8,8))
    for i in range(No[1]):
        x,y = ind2sub(MAZE.shape,i)
        C_target[i,0] = -math.sqrt((x-END_COORD[0])**2 + (y-END_COORD[1])**2)
        K[x,y] = C_target[i,0]

    # C_target = np.zeros((Ns[0],1))
    # C_target[sub2ind(MAZE.shape,END_COORD),0] = 10
    C_.append(C_target)

    layer = mdp_layer()
    layer.T = 20
    layer.options.T_horizon = 4

    layer.A_ = A_
    layer.a_ = a_
    layer.B_ = B_
    layer.C_ = C_
    layer.D_ = D_
    layer.U_ = U_

    layer.s = START
    print("Done !")
    return layer,MAZE,START_COORD,END_COORD,u

def mazeplot(maze,start_position,current_position,belief_matrix,target_position,TT,belief=False):
    maze = maze.T
    maze_shape = maze.shape
    sy = 640
    by = (int)(sy/maze.shape[0])
    sx = int(sy*(float(maze_shape[1])/float(maze_shape[0])))
    bx = (int)(sx/maze.shape[1])

    colors =  ["red", "green", "blue", "yellow","purple", "orange"]
    cold_color = np.array([0.1, 0.1, 1,0.1])
    hot_color = np.array([1, 0.1, 0.1,0.8])
    maze_img = Image.new("RGB",(sx,sy),"white")
    draw = ImageDraw.Draw(maze_img,"RGBA")

    def lerp(a, b, t):
        return a*(1 - t) + b*t

    

    def sub2ind(array_shape, coords):
        cols = coords[1]
        rows = coords[0]
        return cols*array_shape[0] + rows
    
    def ind2sub(array_shape, ind):
        # for 2 d arrays only
        rows = ind//(array_shape[0])
        cols = ind%(array_shape[0])
        return cols,rows


    if (belief):
        pass
    else :
        # DRAW PERCEPTION OF THE MAZE
        perception_matrix = normalize(lay.a_[0],0)
        for k in range((perception_matrix.shape[1])) :
            values = perception_matrix[:,k]
            indices = ind2sub(maze.shape,k)
            position = [bx/2.0 + indices[0]*bx*1.0 ,by/2.0 + indices[1]*by*1.0]
            rectangle_up = (position[0]-bx/2.0,position[1]-by/2.0)
            rectangle_down = (position[0]+bx/2.0,position[1]+by/2.0)
            color = np.array([1, 1, 1])*values[0]
            color = tuple((color*255).astype(np.int))
            draw.rectangle((rectangle_up,rectangle_down),fill=color)

    # DRAW EXPECTED STATES
    for state in range(belief_matrix.shape[0]):
        current_state_coord = ind2sub(maze.shape,state)
        position = [bx/2.0 + current_state_coord[1]*bx*1.0 ,by/2.0 + current_state_coord[0]*by*1.0]
        
        factor  = belief_matrix[state,TT]*1.7
        rectangle_up = (position[0]-factor*bx/2.0,position[1]-factor*by/2.0)
        rectangle_down = (position[0]+factor*bx/2.0,position[1]+factor*by/2.0)

        color = (lerp(cold_color,hot_color,belief_matrix[state,TT]))
        color = tuple((color*255).astype(np.int))
        
        color="blue"
        draw.rectangle((rectangle_up,rectangle_down),fill=color,outline="black")

    if (belief):
        # DRAW BELIEVED MAZE
        perception_matrix = normalize(lay.a_[0],0)
        for k in range((perception_matrix.shape[1])) :
            values = perception_matrix[:,k]
            indices = ind2sub(maze.shape,k)
            position = [ by/2.0 + indices[1]*by*1.0,bx/2.0 + indices[0]*bx*1.0]

            color = np.array([1, 1, 1])*values[0]
            color = tuple((color*255).astype(np.int))
            draw.line(((position[0]-bx/2.0,position[1]-by/2.0),(position[0]+bx/2.0,position[1]+by/2.0)),width = 10,fill=color)
            draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color)

            draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]+by/2.0)),width = 10,fill=color)
            draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]-bx/2.0,position[1]-by/2.0)),width = 10,fill=color)
            draw.line(((position[0]+bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color)
            draw.line(((position[0]-bx/2.0,position[1]-by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color)

    else :
        # DRAW ACTUAL MAZE
        for y in range(maze_shape[0]):
            for x in range (maze_shape[1]):
                position = [bx/2.0 + x*bx*1.0 ,by/2.0 + y*by*1.0]

                if (maze[x,y]>0.5) :
                    color = "black"
                    draw.line(((position[0]-bx/2.0,position[1]-by/2.0),(position[0]+bx/2.0,position[1]+by/2.0)),width = 10,fill=color)
                    draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color)

                    draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]+by/2.0)),width = 10,fill=color)
                    draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]-bx/2.0,position[1]-by/2.0)),width = 10,fill=color)
                    draw.line(((position[0]+bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color)
                    draw.line(((position[0]-bx/2.0,position[1]-by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color)

    # DRAW TARGET
    color = "red"
    position = [bx/2.0 + target_position[1]*bx*1.0 ,by/2.0 + target_position[0]*by*1.0]
    draw.line(((position[0]-bx/2.0,position[1]-by/2.0),(position[0]+bx/2.0,position[1]+by/2.0)),width = 10,fill=color)
    draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color)

    # DRAW BEGINNING
    color = "green"
    position = [bx/2.0 + start_position[1]*bx*1.0 ,by/2.0 + start_position[0]*by*1.0]
    draw.line(((position[0]-bx/2.0,position[1]-by/2.0),(position[0]+bx/2.0,position[1]+by/2.0)),width = 10,fill=color)
    draw.line(((position[0]-bx/2.0,position[1]+by/2.0),(position[0]+bx/2.0,position[1]-by/2.0)),width = 10,fill=color)

    # DRAW CURRENT
    current_position = list(ind2sub(maze.shape,current_position))
    color = "purple"
    shift = 0.1*bx
    position = [bx/2.0 + current_position[1]*bx*1.0 ,by/2.0 + current_position[0]*by*1.0 + shift]
    draw.line(((position[0]-bx/2.0 + shift ,position[1]-by/2.0),(position[0]+bx/2.0 - shift,position[1]-by/2.0)),width = 10,fill=color)
    draw.line(((position[0],position[1]),(position[0]+bx/2.0 - shift,position[1]-by/2.0)),width = 10,fill=color)
    draw.line(((position[0],position[1]),(position[0]-bx/2.0 + shift,position[1]-by/2.0)),width = 10,fill=color)

    #maze_img.save('outal.png',"PNG")
    return maze_img
            


if (__name__ == "__main__"):
    lay,maze,st,en,u = maze_model()
    lay.prep_trial()

    belief = True
    K = 20

    cnt = 0
    for container in lay.run_generator(K):
        cnt += 1
        imL = []
        for t in range(lay.T):
            for i in range(lay.T):
                imL.append(mazeplot(maze,st,container.s[:,t],container.state_expectation[t],en,i,belief))
        path = 'gif_test' + str(cnt) + '.gif'
        print("saving to " + path)
        imL[0].save(path,save_all=True,append_images=imL[1:],duration=50,loop = 0)
        print(container.trees[3].to_string())