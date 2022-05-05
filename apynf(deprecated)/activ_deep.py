# -*- coding: utf-8 -*-
"""
Created on Tue Jul 7 11:06 2021

@author: Côme ANNICCHIARICO(come.annicchiarico@mines-paristech.fr), adaptation of the work of :

%% Step by step introduction to building and using active inference models

% Supplementary Code for: A Step-by-Step Tutorial on Active Inference Modelling and its 
% Application to Empirical Data

% By: Ryan Smith, Karl J. Friston, Christopher J. Whyte
(MATLAB Script)
https://github.com/rssmith33/Active-Inference-Tutorial-Scripts/blob/main/Step_by_Step_AI_Guide.m


AND 

Towards a computational (neuro)phenomenology of mental action: modelling
meta-awareness and attentional control with deep-parametric active inference (2021)

Lars Sandved-Smith  (lars.sandvedsmith@gmail.com)
Casper Hesp  (c.hesp@uva.nl)
Jérémie Mattout  (jeremie.mattout@inserm.fr)
Karl Friston (k.friston@ucl.ac.uk)
Antoine Lutz (antoine.lutz@inserm.fr)
Maxwell J. D. Ramstead (maxwell.ramstead@mcgill.ca)


------------------------------------------------------------------------------------------------------
Generic multilayered model. Uses layers to form a modular brain model.

Long term objective --> A comprehensive class able to simulate both "emotional/attentional" cognitive levels AND contextual levels.


"""
import os,sys
from mdp_layer import mdp_layer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random as r
import time 
import numpy as np
import matplotlib.pyplot as plt
from function_toolbox import normalize,softmax,nat_log,precision_weight
from function_toolbox import spm_wnorm,cell_md_dot,md_dot, spm_cross,spm_KL_dir,spm_psi, spm_dot
from function_toolbox import G_epistemic_value
from plotting_toolbox import basic_autoplot
from miscellaneous_toolbox import isField
from explore_exploit_model import explore_exploit_model


class deep_model:
    def __init__(self,shape_):
        """ For now , only 1 dimensionnal shapes are supported"""
        assert (len(shape_)<=1), "Shapes of dimension > 1 are not yet supported . (input dim = " + str(len(shape_)) + " )"
        self.shape = shape_
        self.layer_grid = []
        for i in range(self.shape[0]):
            layer = mdp_layer()
            layer.name = "layer_" + str(i)
            self.layer_grid.append(layer)

    def link_layers(self):
        for i in range(self.shape[0]):
            layer = self.layer_grid[i]
            if (i > 0):
                layer.child = self.layer_grid[i-1]
            if (i < self.shape[0]-1):
                layer.parent = self.layer_grid[i+1]


    def run(self):
        for layer in self.layer_grid:
            layer.prep_trial()
        

A = deep_model((5,))
A.link_layers()
for i in range(A.shape[0]) :
    if (A.layer_grid[i].parent) :
        print(A.layer_grid[i].name + " 's parent is " +  A.layer_grid[i].parent.name)
    if (A.layer_grid[i].child) :
        print(A.layer_grid[i].name + " 's child is " +  A.layer_grid[i].child.name)
    print('-----')