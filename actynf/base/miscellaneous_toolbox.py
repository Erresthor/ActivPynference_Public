# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:57:40 2021

@author: cjsan

A set of very basic functions
"""
import numpy as np
import random

def isField(x):
    return not(isNone(x))

def clamp(x,lower,upper): 
    if (x>upper):
        x = upper
    elif (x<lower) :
        x = lower
    return x

def isNone(inp): 
    """ 
    Binary classificition, returns bool. 
    If all but one elemements of a sequence are None, it will be classified as << not None >> 
    """
    isnone = True

    if (type(inp)==list)or(type(inp)==tuple):
        for i in range(len(inp)):
            isnone = isnone and isNone(inp[i])
    
    elif (type(inp)==np.ndarray):
        for element in inp :
            isnone = isnone and isNone(element)
    
    else :
        isnone = (inp == None)

    return isnone

def listify(object):
    if (type(object)==list):
        return object
    else :
        return [object]

def flexible_copy(X):
    """ Replace this by deepcopy ?"""
    if (type(X)==list):
        new_list = []
        for i in range(len(X)):
            new_list.append(flexible_copy(X[i]))
        return new_list
    elif (type(X)==np.ndarray):
        return np.copy(X)
    else :
        return X

def flatten_last_n_dimensions(n,array):
    to_this_shape = array.shape[:array.ndim-n]+(-1,)
    # print(to_this_shape)
    return np.reshape(array,to_this_shape,order="C")

def flexible_toString(object,rounder=2):
    if (type(object)==list):
        message = "["
        for i in range(len(object)):
            message = message + flexible_toString(object[i],rounder=rounder)
            if (i<len(object)-1):
                message += ",\n"
        message += "]"
    elif (type(object)==np.ndarray):
        d = object.dtype 
        isInt = (d==np.int)or(d==np.int8)or(d==np.int16)or(d==np.int32)
        if not(isInt):
            message = str(np.round(object,2))
        else : 
            message = str(object)
    else :
        message = str(object)
    return(message)

def flexible_print(object,round = 2):
    print(flexible_toString(object,rounder=round))

def index_to_dist(index,array_example):
    returner = np.zeros(array_example.shape)
    returner[index] = 1
    return returner

def dist_to_index(array,axis=0):
    """ Return the index of the maximum along the given axis"""
    return np.argmax(array,axis=axis)

def flexible_to_list(X):
    if (type(X)==list):
        new_list = []
        for i in range(len(X)):
            new_list.append(flexible_to_list(X[i]))
        return new_list
    elif (type(X)==np.ndarray):
        return X.tolist()
    elif not(isField(X)) :
        return []

def flexible_to_array(X,dims_in_list = 1):
    """ dims_in_list indicate how many dimensions will remain as a list. For most Active Inference schemes, modalities and factors are specified using lists."""
    if (type(X)==list):
        if (X==[]):
            return None
        print(dims_in_list)
        if (dims_in_list>0):
            new_list = []
            for i in range(len(X)):
                new_list.append(flexible_to_array(X[i],dims_in_list-1))
            return new_list
        else :
            return np.array(X)
    elif (type(X)==np.ndarray):
        return X
    else :
        return X

def smooth_1D_array(arr,smooth_size = 5):
    N = arr.shape[0]
    smoothed_one = np.zeros((N,))
    for i in range(N):
        cnt = 0
        tot = 0
        for k in range(i-smooth_size,i+smooth_size+1,1):
            if ((k>=0)and(k<N)):
                cnt = cnt + 1
                tot = tot + arr[k]
        smoothed_one[i] = tot/cnt
    return smoothed_one

def sliding_window_mean(array_input,window_size = 5):
    list_output = np.zeros(array_input.shape)
    N = array_input.shape[0]
    for trial in range(N):
        mean_value = 0
        counter = 0
        for k in range(trial - window_size,trial + window_size + 1):
            if(k>=0):
                try :
                    mean_value += array_input[k]
                    counter += 1
                except :
                    a = 0
                    #Nothing lol
        list_output[trial] = mean_value/counter
    return list_output

def numpy_array_of_Nan(shape):
    a = np.empty(shape)
    a[:] = np.nan
    return a

def nan_in_array(array):
    return np.isnan(array).any()

def sample_distribution(distribution, N=1):
    sum_of_dist = np.sum(distribution)
    assert abs(sum_of_dist-1.0) < 1e-10, "A probabilistic distribution should sum to 1 (sums to " + str(sum_of_dist) + " )"
    
    L = []
    for k in range(len(N)):
        np.argwhere(random.random() <= np.cumsum(distribution,axis=0))[0]