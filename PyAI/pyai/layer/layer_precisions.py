# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:44:10 2021

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
This class describes the precisions for a given layer
"""
import os,sys
from enum import Enum
import random as r
import time 
import numpy as np
import matplotlib.pyplot as plt

from ..base.miscellaneous_toolbox import isField

def init_precisions(layer_input):
    T = layer_input.T
    Ni = layer_input.options.Ni
    Nmod = layer_input.Nmod
    Nf = layer_input.Nf
    
    # Initialize precisions
    beta_mat = np.full((1,T),-1.0)
    if isField(layer_input.precisions.policy.BETA) :
        beta_mat[layer_input.precisions.policy.BETA>0] = layer_input.precisions.policy.BETA[layer_input.precisions.policy.BETA>0]
    layer_input.precisions.policy.BETA = np.copy(beta_mat)
    prior_mat =  np.full((1,),-1.0)
    if isField(layer_input.precisions.policy.prior) :
        prior_mat[layer_input.precisions.policy.prior>0] = layer_input.precisions.policy.prior[layer_input.precisions.policy.prior>0]
    layer_input.precisions.policy.prior = np.copy(prior_mat)
    layer_input.precisions.policy.beta = np.ones((1,T))
    layer_input.precisions.policy.beta_n = np.ones((1,Ni*T))
    
    

    beta_mat = np.full((Nmod,T),-1.0)
    if isField(layer_input.precisions.A.BETA) :
        beta_mat[layer_input.precisions.A.BETA>0] = layer_input.precisions.A.BETA[layer_input.precisions.A.BETA>0]
    layer_input.precisions.A.BETA = np.copy(beta_mat)
    prior_mat =  np.full((Nmod,),-1.0)
    if isField(layer_input.precisions.A.prior) :
        prior_mat[layer_input.precisions.A.prior>0] = layer_input.precisions.A.prior[layer_input.precisions.A.prior>0]
    layer_input.precisions.A.prior = np.copy(prior_mat)
    layer_input.precisions.A.beta = np.ones((Nmod,T))
    layer_input.precisions.A.beta_n = np.ones((Nmod,Ni*T))


    beta_mat = np.full((Nf,T),-1.0)
    if isField(layer_input.precisions.B.BETA) :
        beta_mat[layer_input.precisions.B.BETA>0] = layer_input.precisions.B.BETA[layer_input.precisions.B.BETA>0]
    layer_input.precisions.B.BETA = np.copy(beta_mat)
    prior_mat =  np.full((Nf,),-1.0)
    if isField(layer_input.precisions.B.prior) :
        prior_mat[layer_input.precisions.B.prior>0] = layer_input.precisions.B.prior[layer_input.precisions.B.prior>0]
    layer_input.precisions.B.prior = np.copy(prior_mat)
    layer_input.precisions.B.beta = np.ones((Nf,T))
    layer_input.precisions.B.beta_n = np.ones((Nf,Ni*T))


    beta_mat = np.full((Nmod,T),-1.0)
    if isField(layer_input.precisions.C.BETA) :
        beta_mat[layer_input.precisions.C.BETA>0] = layer_input.precisions.C.BETA[layer_input.precisions.C.BETA>0]
    layer_input.precisions.C.BETA = np.copy(beta_mat)
    prior_mat =  np.full((Nmod,),-1.0)
    if isField(layer_input.precisions.C.prior) :
        prior_mat[layer_input.precisions.C.prior>0] = layer_input.precisions.C.prior[layer_input.precisions.C.prior>0]
    layer_input.precisions.C.prior = np.copy(prior_mat)
    layer_input.precisions.C.beta = np.ones((Nmod,T))
    layer_input.precisions.C.beta_n = np.ones((Nmod,Ni*T))


    beta_mat = np.full((Nf,T),-1.0)
    if isField(layer_input.precisions.D.BETA) :
        beta_mat[layer_input.precisions.D.BETA>0] = layer_input.precisions.D.BETA[layer_input.precisions.D.BETA>0]
    layer_input.precisions.D.BETA = np.copy(beta_mat)
    prior_mat =  np.full((Nf,),-1.0)
    if isField(layer_input.precisions.D.prior) :
        prior_mat[layer_input.precisions.D.prior>0] = layer_input.precisions.D.prior[layer_input.precisions.D.prior>0]
    layer_input.precisions.D.prior = np.copy(prior_mat)
    layer_input.precisions.D.beta = np.ones((Nf,T))
    layer_input.precisions.D.beta_n = np.ones((Nf,Ni*T))


    beta_mat = np.full((1,T),-1.0)
    if isField(layer_input.precisions.E.BETA) :
        beta_mat[layer_input.precisions.E.BETA>0] = layer_input.precisions.E.BETA[layer_input.precisions.E.BETA>0]
    layer_input.precisions.E.BETA = np.copy(beta_mat)
    prior_mat =  np.full((1,),-1.0)
    if isField(layer_input.precisions.E.prior) :
        prior_mat[layer_input.precisions.E.prior>0] = layer_input.precisions.E.prior[layer_input.precisions.E.prior>0]
    layer_input.precisions.E.prior = np.copy(prior_mat)
    layer_input.precisions.E.BETA = np.ones((1,T))
    layer_input.precisions.E.beta = np.ones((1,T))
    layer_input.precisions.E.beta_n = np.ones((1,Ni*T))

class mdp_layer_block_precision :
    def __init__(self,id_,inherited_,correspondance_matrix_=None,prior_=None,to_infer_=None):
        self.id = id_                                               # Denomination 
        self.inherited = inherited_                                 # This block precision is induced by upper states ? (yes / no)
        self.correspondance_matrix = correspondance_matrix_         # SUM(Outcome(lvl+1)*correspondance_matrix) = new_prec
        self.prior = prior_                                         # If we don't depend on a parent node, what is the prior
        self.to_infer = to_infer_                                   # Should we try to infer its value ?

        self.BETA = None                                      # Array, gt values --> Size = (Modality / Factor)  x  T  [-- x Np ? --]
        self.beta = None                                      # Array, infered values --> Size = (Modality / Factor)  x  T  [-- x Np ? --]
        self.beta_n = None                                    # Array, infered values --> Size = Ni  x  (Modality / Factor)  x  T  [-- x Np ? --]
    
    def fill_empty_BETAs(self,t,fillValue=-1):
        assert (isField(self.BETA)),"BETA " + self.id+ "  not implemented ... Please check that precisions are initialized before setting values."
        dim = self.BETA.shape[0] 
        for i in range(dim):
            if (self.BETA[i,t]<0):
                if (fillValue >= 0):
                    self.BETA[i,t] = fillValue
                else :
                    self.BETA[i,t] = self.prior[i]

    def fill_empty_priors(self,fillValue):
        assert (isField(self.prior)),"prior " + self.id+ "  not implemented ... Please check that precisions are initialized before setting values."
        dim = self.prior.shape[0] 
        for i in range(dim):
            if (self.prior[i]<0):
                self.prior[i] = fillValue

class mdp_layer_precisions :
    def __init__(self):
        self.policy = mdp_layer_block_precision("POL",False)
        self.A = mdp_layer_block_precision("A",False)
        self.B = mdp_layer_block_precision("B",False)
        self.C = mdp_layer_block_precision("C",False)
        self.D = mdp_layer_block_precision("D",False)
        self.E = mdp_layer_block_precision("E",False)

    def count_inherited_precisions(self):
        count = 0 
        if (self.A.inherited) :
            count = count + 1
        if (self.B.inherited) :
            count = count + 1
        if (self.C.inherited) :
            count = count + 1
        if (self.D.inherited) :
            count = count + 1
        if (self.E.inherited) :
            count = count + 1
        if (self.policy.inherited) :
            count = count + 1
        return count

    def fill_all_empty_BETAs(self,t,fillValue = -1):
        self.policy.fill_empty_BETAs(fillValue,t)
        self.A.fill_empty_BETAs(t,fillValue)
        self.B.fill_empty_BETAs(t,fillValue)
        self.C.fill_empty_BETAs(t,fillValue)
        self.D.fill_empty_BETAs(t,fillValue)
        self.E.fill_empty_BETAs(t,fillValue)

    
