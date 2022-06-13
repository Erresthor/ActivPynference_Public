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
from .miscellaneous_toolbox import isField
from .function_toolbox import normalize

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
    if not(isField(listmat1)) or not(isField(listmat2)):
        try :
            n = len(listmat1)
        except :
            try :
                n = len(listmat2)
            except :
                return None
        return [0 for i in range(n)]
    distance_list = []
    for k in range(len(listmat1)):
        distance_list.append(matrix_distance(listmat1[k],listmat2[k],metric=metric))
    return distance_list


def uncertainty(distribution,factor = 1/2):
    """ For factor in [0,1[, a measure of how "dispersed" the information is in subject beliefs."""
    
    K = distribution.shape[0]
    normalized_distribution = normalize(distribution)
    sorted_dist = -np.sort(-normalized_distribution)

    def find_point_in(y,data):
        """ data is an decreasing set of probability density"""
        x = -1
        K = data.shape[0]
        for k in range (K-1):
            x += 1
            if (data[x]>y) and (data[x+1]<=y):
                return(x)
        return K

    def uncertainty_estimator(distribution,epsilon= 0):
        """ Which proportion of the distribution space accounts for probabilities > 1/N - epsilon ?"""
        N = distribution.shape[0]
        fixed_point_y = min(1/N - epsilon,N)
        assert fixed_point_y >= 0, "uncertainty parameter should be > 1 / distribution space size, instead of " + str(fixed_point_y)

        fixed_point_x = find_point_in(fixed_point_y,distribution)
        return fixed_point_x/N,fixed_point_x,fixed_point_y
    
    epsilon = factor*(1.0/K)
    rating,x,y =  uncertainty_estimator(sorted_dist,epsilon)
    return rating

def multidimensionnal_uncertainty(matrix,factor = .5):
    if (matrix.ndim==1):
        return uncertainty(matrix,factor)
    else : 
        uncertainty_matrix = np.zeros(matrix.shape[:-1])
        it = np.nditer(matrix[...,0], flags=['multi_index'])
        for x in it:
            vector_slicer = it.multi_index + (slice(None),)
            vector = (matrix[vector_slicer])
            vector_uncertainty = uncertainty(vector,factor)
            uncertainty_matrix[it.multi_index] = vector_uncertainty
        return (uncertainty_matrix)

def calculate_uncertainty(matrix,factor = .5):
    if (type(matrix)==np.ndarray):
        return multidimensionnal_uncertainty(matrix,factor)
    elif (type(matrix)==list) : 
        matlist = []
        for k in range(len(matrix)):
            matlist.append(calculate_uncertainty(matrix[k],factor))
        return (matlist)
    else :
        raise Exception("Inputs for calculate_uncertainty should be either numpy arrays or list of numpy arrays.")

def mean_uncertainty(matlist,factor=.5):
    uncertainty_matlist = calculate_uncertainty(matlist,factor)
    if (type(uncertainty_matlist)==np.ndarray):
        return np.mean(uncertainty_matlist)
    elif (type(uncertainty_matlist)==list) : 
        matlist = []
        for k in range(len(uncertainty_matlist)):
            matlist.append(np.mean(uncertainty_matlist[k]))
        return (matlist)
    else :
        raise Exception("Inputs for mean_uncertainty should be either numpy arrays or list of numpy arrays.")

# a = np.random.random((9,9,8,2))
# b = np.random.random((9,9,8,2))

# print(matrix_distance(a,b,"1"))
# print(matrix_distance(a,b,"2"))
# print(matrix_distance(a,b,"1200"))
# print(matrix_distance(a,b,"inf"))
# print(9*9*8*2)

def argmean(matrix,axis=0):
    """For spacially coherent matrices.
    Returns the weighted sum of indices"""
    it = np.nditer(matrix,flags=['multi_index'])
    indices_matrix = np.zeros(matrix.shape)
    for x in it:
        ind = (it.multi_index)
        indices_matrix[ind] = ind[axis]
    # # We want indices array to be the same shape as matrix[i,...]
    # print(indices_array)
    return np.sum(matrix*indices_matrix,axis)

if __name__=="__main__":
    print(">:(")