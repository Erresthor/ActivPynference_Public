# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:57:40 2021

@author: cjsan
"""
import numpy as np
import pandas as pd

def isField(x):
    return not(isNone(x))

def clamp(x,lower,upper): 
    if (x>upper):
        x = upper
    elif (x<lower) :
        x = lower
    return x

def isNone(inp): 
    """ Binary classificition, returns bool. If all but one elemements of a sequence are None, it will be classified as << not None >> """
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


def flexible_copy(X):
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
    return np.reshape(array,array.shape[:array.ndim-n]+(-1,))

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

if __name__ == "__main__":
    A = np.round(np.random.rand(5,6,3,3),1)
    B = np.round(np.random.rand(5,6,3,3),1)
    
    Alist = A.tolist()
    Blist = B.tolist()
    
    data = {'first_column':["A","B"],
            'flattened':[Alist,Blist]}
    
    df =pd.DataFrame(data)
    print(df)
    
    list = (flexible_to_array([Alist,Blist]))
    print( list[0].shape)