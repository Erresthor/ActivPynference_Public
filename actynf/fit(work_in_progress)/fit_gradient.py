# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15th 2022

Use a gradient based algorithm to fit an [observations,observed_actions] dataset to an active inference paradigm

Work in progress ^^"


@author: cjsan
"""
import numpy as np
import random
from PIL import Image, ImageDraw
import sys,os,inspect
import math
import matplotlib.pyplot as plt
import itertools
import random

from ..base.miscellaneous_toolbox import flexible_copy , isField
from ..base.function_toolbox import normalize
from ..base.function_toolbox import spm_dot,spm_kron
from ..base.plotting_toolbox import multi_matrix_plot
from ..base.file_toolbox import load_flexible,save_flexible
from ..layer_old.mdp_layer import mdp_layer
from ..layer_old.layer_postrun import evaluate_run

class parameters :
    def __init__(self,a,b,c,d,e,u,o):
        self.a = a 
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        
        self.u = u
        self.o = o
        
        self.Ns = []
        for k in range(len(self.b)):
            self.Ns.append(self.b[k].shape[0])
        
        self.No = []
        for k in range(len(self.a)):
            self.No.append(self.a[k].shape[0])





    def parameter_array(self):
        array = []



        return array


def w_loss(u_tofit,u_layer):
    layer_acts = u_layer
    data_acts = u_tofit

    counter = 0.0
    total = 0.0
    for fac in range(layer_acts.shape[0]):
        for t in range(layer_acts.shape[1]):
            if (abs(layer_acts[fac,t]-data_acts[fac,t])>1e-5):
                total = total + 1.0
            counter = counter + 1
    acts_mse = total/counter
    return acts_mse

def evaluate_parameters(T,parameters,to_be_fitted_layer):
    return evaluate_layer(initialize_layer(T,a,b,c,d,e,u,o), to_be_fitted_layer)

def evaluate_layer(lay,to_be_fitted_layer):
    return w_loss(to_be_fitted_layer, lay)

def initialize_layer(T,a,b,c,d,e,u,o):
    layer = mdp_layer()
    layer.options.T_horizon = 1
    layer.T = T

    layer.A_ = a
    layer.a_ = a

    layer.B_ = b
    layer.b_ = b

    layer.C_ = c

    layer.d_ = d
    layer.D_ = d

    layer.U_ = u

    return layer

def generate_random_parameters(layer_dims,based_on) :
    Ns = layer_dims[0]
    No = layer_dims[1]
    Nu = layer_dims[2]
    T = layer_dims[3]
    u = layer_dims[4]
    o = layer_dims[5]

    a = []
    for mod in range(len(No)):
        a.append(normalize(np.random.random((No[mod],)+tuple(Ns))))
    b = []
    for fac in range(len(Ns)):
        #b.append(normalize(np.random.random((Ns[fac],Ns[fac],Nu[fac]))))
        b[-1]=(based_on.b_[fac]) 
    c = []
    for mod in range(len(No)):
        #c.append(normalize(np.random.random((No[mod],1))))
        c.append(based_on.C_[mod])
    d = [] 
    for fac in range(len(Ns)):
        d.append(normalize(np.random.random((Ns[fac]))))

    return parameters(a, b, c, d, e, u, o)


def gradient(parameters,to_fit):
    parameter_array








