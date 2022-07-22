# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:58:11 2021

@author: cjsan
"""
import os,sys
from enum import Enum
import random as r
import time 
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import pickle

def root_path():
    return os.path.abspath(os.sep)

def save_flexible(obj,to_path):
    if not os.path.exists(os.path.dirname(to_path)):
        try:
            os.makedirs(os.path.dirname(to_path))
        except OSError as exc: # Guard against race condition
            raise
            if exc.errno != errno.EEXIST:
                raise
    with (open(to_path,'wb')) as f:
        pickle.dump(obj,f)
        # if (type(obj)==np.ndarray):
        #     np.save(to_path,object,True,True)
        # else :
        #     pickle.dump(obj,f)

def load_matrix(to_path):
    with(open(to_path,'rb')) as f :
        obj = np.load(to_path)
    return obj

def load_pickled(to_path):
    with open(to_path,'rb') as f:
        obj = pickle.load(f)
    return obj

def load_flexible(to_path):
    try :
        obj = load_matrix(to_path)
    except :
        obj = load_pickled(to_path)
    return obj

def filename_in_files(list_files,my_filename):
    for filename in (list_files):
        if my_filename in filename :
            return True
    return False