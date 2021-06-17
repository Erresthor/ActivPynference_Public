# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:57:40 2021

@author: cjsan
"""

import numpy as np

def isField(x):
    if (type(x)==list) :
        return True
    if (type(x)==np.ndarray):
        return True
    if (x==None) :
        return False
    return True

def clamp(x,lower,upper):
    if (x>upper):
        x = upper
    elif (x<lower) :
        x = lower
    return x

