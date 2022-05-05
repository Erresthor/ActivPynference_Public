 # -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:44:10 2021

@author: Côme ANNICCHIARICO(annicchiarico.come@gmail.com)

Take a layer, and save its run components to a normalized file.
"""
import os,sys
from enum import Enum
import random as r
import time 
import numpy as np
import matplotlib.pyplot as plt

from ..base.miscellaneous_toolbox import isField,flexible_copy
from ..base.function_toolbox import normalize , spm_kron, spm_wnorm, nat_log , spm_psi, softmax , spm_cross
from ..base.function_toolbox import spm_kron,spm_margin,spm_dekron
from ..base.file_toolbox import load_flexible,save_flexible
from .parameters.policy_method import Policy_method
from .layer_postrun import evaluate_run
#from pynf_functions import *


def f(name,k=None,supp=None) :
    strk = ""
    if (k!=None) :
        strk = "_" + '{:06d}'.format(k)
    
    strsupp = ""
    if(supp !=None):
        strsupp = "_" + str(supp)
    

    file_format = ".txt"
    basepath = str(name) + strsupp + strk + file_format
    return basepath

def load_layer_sumup(from_path):
        return (load_flexible(from_path))

class layer_exp_sumup :
    def __init__(self,k,savepath):
        self.savepath = savepath
        self.k = k
        self.ticklist = []
    
    def add_layer_tick_sumup(self,tick_sumup):
        self.ticklist.append(tick_sumup)
    
    def save_layer_exp_sumup(self):
        basepath = os.path.join(self.savepath,f(self.ticklist[0].name,k=self.k))
        save_flexible(self, basepath)
    
    def get_final_state(self):
        return self.ticklist[-1]

class layer_tick_sumup :
    """ A class stocking what a layer looked like at a given timestamp t. """
    def __init__(self,layer,k=None,supplex=None,t = 0):
        self.k = k
        self.name = layer.name
        self.t = layer.t # this is the time AFTER the tick is done. Therefore, the last possible recorded time for any experience is layer.T.
        self.T = layer.T

        # Results
        self.Q = flexible_copy(layer.Q)
        self.o = flexible_copy(layer.o)
        self.u = flexible_copy(layer.u)
        self.s = flexible_copy(layer.s)    
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

    def save(self,to_path):
        save_flexible(self, to_path)
    
    def save_base(self):
        self.save(self.basepath)

    def load_base(self):
        return load_layer_sumup(self.basepath)
    