from audioop import avg
from unicodedata import unidata_version
import numpy as np
import matplotlib.pyplot as plt
import pandas
from matplotlib.animation import FuncAnimation
from pyai.base.matrix_functions import matrix_distance_list
from pyai.base.function_toolbox import normalize,spm_KL_dir,KL_test,KL_div_variant

# A = np.array([[1,0,0,0,0],
#               [0,1,0,0,0],
#               [0,0,1,0,0],
#               [0,0,0,1,0],
#               [0,0,0,0,1]])

# B = np.array([[0,0,1,0,1],
#               [0,0,0,1,0],
#               [0,0,0,0,0],
#               [0,1,0,0,0],
#               [1,0,0,0,0]])
# print(normalize(A))
# print(normalize(B))
# print(matrix_distance_list([normalize(A)],[normalize(B)],metric='2'))
# print(np.sqrt(10)) 

def my_KL_dir(x,y):
    returner = 0
    for k in range(x.shape[0]):
        returner += x[k]*np.log((x[k]+1./32)/(y[k]+1./32))
    return returner

A = np.array([0.25,0.5,0.25,0])
B = np.array([0.25,0.5,0.25,0])
C = np.array([0,0.25,0.5,0.25])

print(spm_KL_dir(A,B),my_KL_dir(A,B),KL_test(A,B),KL_div_variant(A,B))

print(spm_KL_dir(A,C),my_KL_dir(A,C),KL_test(A,C),KL_div_variant(A,C))

print(spm_KL_dir(C,A),my_KL_dir(C,A),KL_test(C,A),KL_div_variant(C,A))

