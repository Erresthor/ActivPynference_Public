from audioop import avg
from unicodedata import unidata_version
import numpy as np
import matplotlib.pyplot as plt
import pandas
from matplotlib.animation import FuncAnimation
from pyai.base.matrix_functions import matrix_distance_list
from pyai.base.function_toolbox import normalize,spm_KL_dir,KL_test,KL_div_variant
from pyai.base.matrix_functions import calculate_uncertainty,mean_uncertainty
from pyai.base.plotting_toolbox import multi_matrix_plot
import sys,os 
from pyai.model.metrics import avg_kl_dir
from pyai.base.matrix_dist_operations import generate_normal_dist_along_matrix

n = 5


def return_B():
    nu = 5
    npoubelle = 2
    Ns = [5]
    pb = 1
    B_mental_states = np.zeros((Ns[0],Ns[0],nu+npoubelle))

    # Line = where we're going
    # Column = where we're from
    B_mental_states[:,:,0] = np.array([ [1  ,1  ,1,1,1],         # Try to move to terrible state from others
                                        [0  ,0  ,0,0,0],
                                        [0  ,0  ,0,0,0],
                                        [0  ,0  ,0,0,0],
                                        [0  ,0  ,0,0,0]])

    B_mental_states[:,:,1] = np.array([[1-pb,0  ,0  ,0  ,0  ],         # Try to move to neutral state from others
                                        [pb ,1  ,1  ,1  ,1  ],
                                        [0  ,0  ,0  ,0  ,0  ],
                                        [0  ,0  ,0  ,0  ,0  ],
                                        [0  ,0  ,0  ,0  ,0 ]])

    B_mental_states[:,:,2] = np.array([ [1  ,0   ,0  ,0   ,0  ],         # Try to move to good state from others
                                        [0  ,1-pb,0  ,0   ,0  ],
                                        [0  ,pb  ,1  ,1   ,1  ],
                                        [0  ,0   ,0  ,0   ,0  ],
                                        [0  ,0   ,0  ,0   ,0  ]])

    B_mental_states[:,:,3] = np.array([ [1  ,0  ,0   ,0  ,0  ],         # Try to move to target state from others
                                        [0  ,1  ,0   ,0  ,0  ],
                                        [0  ,0  ,1-pb,0  ,0  ],
                                        [0  ,0  ,pb  ,1  ,1  ],
                                        [0  ,0  ,0   ,0  ,0  ]])

    B_mental_states[:,:,4] = np.array([ [1  ,0  ,0  ,0  ,1-pb],         # Try to move to best state from others
                                        [0  ,1  ,0  ,0  ,0  ],
                                        [0  ,0  ,1  ,0  ,0  ],
                                        [0  ,0  ,0  ,1-pb,0  ],
                                        [0  ,0  ,0  ,pb ,pb]])
    for k in range(nu,nu+npoubelle):
        B_mental_states[:,:,k] = normalize(np.random.random((5,5)))
        B_mental_states[:,:,k] = np.eye(5)

    return B_mental_states

dmu = 0
sigma = 4

A = np.eye(n)
normaled_A = generate_normal_dist_along_matrix(A,sigma)
multi_matrix_plot([A,normaled_A],["A","A gaussian"],xlab="States",ylab="Observations",colmap='viridis')
plt.show()

B = return_B()

B = np.zeros((n,n,3))

for k in range(n):
    # GO UP :
    try :
        assert k+1<n,"k"
        B[k+1,k,0] = 1
    except :
        B[k,k,0] = 1

    # GO DOWN :
    try :
        assert k-1>=0,"k"
        B[k-1,k,1] = 1
    except :
        B[k,k,1] = 1
B[:,:,2] = np.eye(n)


normaled_B = generate_normal_dist_along_matrix(B,sigma)
multi_matrix_plot([B,normaled_B],["B","B gaussian"],xlab="States t-1",ylab="States t+1",colmap='viridis')
plt.show()
