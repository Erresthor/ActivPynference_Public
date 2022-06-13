import os,sys
from enum import Enum
import random as r
import time 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..base.miscellaneous_toolbox import isField,flexible_copy,flexible_to_list
from ..base.file_toolbox import load_flexible,save_flexible
#from pynf_functions import *


class ActiveModelSaveContainer():
    # A pandas dataframe would probably make a better alternative and allow for simplified visualization ... Working on it !
    # A lot of matrices with changing dimensions --> HDF5 files might be better for the job
    
    def __init__(self,savepath,layer,trial_counter):
        self.path = savepath
        self.layer_instance = 0
        self.trial = trial_counter
        
        self.t = layer.t # this is the time AFTER the tick is done. Therefore, the last possible recorded time for any experience is layer.T, and the first tick is t = 1.
        self.T = layer.T

        # Results
        self.X = flexible_copy(layer.X)
        self.Q = flexible_copy(layer.Q)
        self.o = flexible_copy(layer.o)
        self.u = flexible_copy(layer.u)
        self.s = flexible_copy(layer.s)  
        self.U_post = flexible_copy(layer.u_posterior)

        # Matrices
        self.a_ = flexible_copy(layer.a_) 
        self.A_ = flexible_copy(layer.A_) 

        self.b_ = flexible_copy(layer.b_) 
        self.B_ = flexible_copy(layer.B_)
 
        self.c_ = flexible_copy(layer.c_)
        self.C_ = flexible_copy(layer.C_)

        self.d_ = flexible_copy(layer.d_)
        self.D_ = flexible_copy(layer.D_)

        self.e_ = flexible_copy(layer.e_)
        self.E_ = flexible_copy(layer.E_)

        self.rt = flexible_copy(layer.rt)
        
        self.FE = layer.FE_dict.copy()

    def as_dict(self):
        return {
            "name":self.path,
            "instance":self.layer_instance,
            "trial":self.trial,
            "T":self.T,
            "Q":flexible_to_list(self.Q),
            "X":flexible_to_list(self.X),
            "o":flexible_to_list(self.o),
            "u":flexible_to_list(self.u),
            "s":flexible_to_list(self.s),
            "Upost":flexible_copy(self.U_post),
            
            "a":flexible_to_list(self.a_),
            "A":flexible_to_list(self.A_),
            
            "b":flexible_to_list(self.b_),
            "B":flexible_to_list(self.B_),
            
            "c":flexible_to_list(self.c_),
            "C":flexible_to_list(self.B_),
            
            "d":flexible_to_list(self.d_),
            "D":flexible_to_list(self.B_),
            
            "e":flexible_to_list(self.e_),
            "E":flexible_to_list(self.E_),
            
            "rt":flexible_to_list(self.rt)
        }
    
    def save(self,to_path):
        save_flexible(self, to_path)

    def quicksave(self):
        self.save(self.path)

    def load_active_model_container(from_path):
        return (load_flexible(from_path))

    def return_dataframe(self):
        return pd.DataFrame.from_dict(data = self.as_dict())