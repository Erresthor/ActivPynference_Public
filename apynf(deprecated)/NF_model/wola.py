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
from base.function_toolbox import normalize,spm_kron,spm_dot
from mdp_layer import mdp_layer



# def custom_kron(A,B,epsilon=1e-2):
#     # A and B should be 2-dimensionnal :
#     if(A.ndim < 2):
#         A = np.expand_dims(A,-1)
#     if(B.ndim < 2):
#         B = np.expand_dims(B,-1)
#     xSize = A.shape[0]*B.shape[0]
#     ySize = A.shape[1]*B.shape[1]
#     ret = np.zeros((xSize,ySize))

#     ind_A = np.argwhere(A>epsilon)
#     ind_B = np.argwhere(B>epsilon)
#     for a in ind_A :
#         for b in ind_B :
#             ret[a[0]*B.shape[0] + b[0],a[1]*B.shape[1] + b[1]] = A[a[0],a[1]]*B[b[0],b[1]]
#     print(ret)
#     return ret


# A = np.array([0.5,0.5,0,0,0])
# B = np.array([0.5,0.5])
# custom_kron(A,B)
# custom_kron(B,A)
# #print(np.reshape(custom_kron(A,B),(4,2)))

# D = [A,B]

# A = np.zeros((5,5,2))

# # A[:,:,0] = np.array([[1,0,0,0],
# #                     [0,1,0,0],
# #                     [0,0,1,1]])

# # A[:,:,1] = np.array([[0.5,0,0,0],
# #                      [0.5,1,0,0],
# #                      [0  ,0,1,1]])
# A_obs_mental =np.zeros((5,5,2))
# A_obs_mental[:,:,0] = np.array([[1,0,0,0,0],
#                                 [0,1,0,0,0],
#                                 [0,0,1,0,0],
#                                 [0,0,0,1,0],
#                                 [0,0,0,0,1]])
# # When distracted, the feedback is modelled as noisy :
# A_obs_mental[:,:,1] = np.array([[0.5 ,0.25,0   ,0   ,0   ],
#                                 [0.5 ,0.5 ,0.25,0   ,0   ],
#                                 [0   ,0.25,0.5 ,0.25,0   ],
#                                 [0   ,0   ,0.25,0.5 ,0.5 ],
#                                 [0   ,0   ,0   ,0.25,0.5]])
# A = [A_obs_mental]

# O = np.array([1,0,0,0,0])

# P = spm_kron(D)
# L = 1
# for i in range(1):
#     L = L* spm_dot(A[0],O)

# print(L.flatten())


# newP = P*L.flatten()
# print(normalize(newP))


# print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
# m = np.array([1,2,3,4,5])
# indexes = (0,4)
# print(m[list(indexes)])

# L = []
# print(L + [1])


a = 23
b = a//15 + 1
c = a%15
print(a,b,c)