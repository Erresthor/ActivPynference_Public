# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:57:40 2021

@author: cjsan
"""

import numpy as np

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


if __name__ == "__main__":
    A = np.round(np.random.rand(5,6,3,3),1)