# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:58:11 2021

@author: cjsan
"""
import numpy as np
from scipy.special import gammaln
from scipy.special import psi
import random as random
import math as m


def matrix_distance(mat1,mat2,metric="2"):
    """evaluate matrix distance between two same dimension matrices."""

    if (metric=="1") :
        return np.sum(np.abs(mat1 - mat2))
            

    elif (metric=="2") :
        return np.power(np.sum(np.power(mat1 - mat2,2)),0.5)

    elif (metric=="inf") :
        return np.max(np.abs(mat1 - mat2))

    elif (metric=="dist"):
        print("to be implemented --> KL Dir simili ?")
    else :
        try :
            m = int(metric)
            return np.power(np.sum(np.power(mat1 - mat2,m)),1/m)
        except :
            return

def matrix_distance_list(listmat1,listmat2,metric="2"):
    distance_list = []
    for k in range(len(listmat1)):
        distance_list.append(matrix_distance(listmat1[k],listmat2[k],metric=metric))
    return distance_list

# a = np.random.random((9,9,8,2))
# b = np.random.random((9,9,8,2))

# print(matrix_distance(a,b,"1"))
# print(matrix_distance(a,b,"2"))
# print(matrix_distance(a,b,"1200"))
# print(matrix_distance(a,b,"inf"))
# print(9*9*8*2)