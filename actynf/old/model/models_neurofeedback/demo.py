import numpy as np
import random
from PIL import Image, ImageDraw
import sys,os,inspect
import math

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from mdp_layer import mdp_layer

a = .98
b = 1 - a
T = 3

A_1 = np.zeros((5,4,2))
A_1[:,:,0] = np.array([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,1,0],
                       [0,0,0,1],
                       [0,0,0,0]])
A_1[:,:,1] = np.array([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,1,0],
                       [0,0,0,0],
                       [0,0,0,1]])

A_2 = np.zeros((3,4,2))
A_2[:,:,0] = np.array([[1,0,0,1],
                       [0,a,b,0],
                       [0,b,a,0]])
A_2[:,:,1] = np.array([[1,0,0,1],
                       [0,b,a,0],
                       [0,a,b,0]])
A = [A_1,A_2]


B_1 = np.zeros((4,4,4))

B_1[:,:,0] = np.array([[1,0,0,1],
                       [0,1,0,0],
                       [0,0,1,0],
                       [0,0,0,0]])
B_1[:,:,1] = np.array([[0,0,0,0],
                       [1,1,0,1],
                       [0,0,1,0],
                       [0,0,0,0]])
B_1[:,:,2] = np.array([[0,0,0,0],
                       [0,1,0,0],
                       [1,0,1,1],
                       [0,0,0,0]])
B_1[:,:,3] = np.array([[0,0,0,0],
                       [0,1,0,0],
                       [0,0,1,0],
                       [1,0,0,1]])
B_2 = np.zeros((2,2,1))
B_2[:,:,0] = np.array([[1,0],
                       [0,1]])
B = [B_1,B_2]


C_1 = np.outer(np.zeros((5,)),np.ones((T,)))
C_2 = np.outer(np.array([0,2,-4]),np.ones((T,)))
C = [C_1,C_2]

d = [np.array([128,1,1,1]),np.array([2,2])]

U = np.array([[0,0],
              [1,0],
              [2,0],
              [3,0]])

layer = mdp_layer()
layer.T = T
layer.Ni = 16
layer.A_ = A
#model.a_ = a_

layer.d_ = d
layer.D_ = d
layer.B_  = B
#    model.b_ = b_

layer.C_ = C

layer.U_ = U

layer.N = 2

layer.s = np.array([0,0])

layer.run()
