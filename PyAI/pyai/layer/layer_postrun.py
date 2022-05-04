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
A method initializing the various sizes used in Active Inference Experiments
"""
import os,sys
from enum import Enum
import random as r
import time 
import numpy as np
import matplotlib.pyplot as plt

from ..base.miscellaneous_toolbox import isField
from ..base.function_toolbox import normalize , spm_kron, spm_wnorm, nat_log , spm_psi, softmax , spm_cross
from ..base.function_toolbox import spm_kron,spm_margin,spm_dekron
from .parameters.policy_method import Policy_method

def evaluate_run(efficacy,run_observations):
    T = efficacy[0].shape[-1]  #Number of timesteps
    return(evaluate_up_to_tau(T, efficacy, run_observations))

def evaluate_timestamp(t,efficacy,obs,optimality_override = None):
    Nmod = len(efficacy)
    evaluation = []

    for modality in range(Nmod):
        evaluation.append(0)

        max_mark = np.max(efficacy[modality][:,t]) # 1D array if everything is
        observed = obs[modality,t]

        points = efficacy[modality][observed,t]
        if (optimality_override!=None):
            try :
                if abs(float(optimality_override[modality][t])) < 0.01 :
                    evaluation[modality] += 1
                else :
                    evaluation[modality] += (float(points)/float(optimality_override[modality][t]))
            except:
                evaluation[modality] += (float(points)/float(max_mark))
        else : 
            evaluation[modality] += (float(points)/float(max_mark))
    return evaluation
    
def evaluate_up_to_tau(tau,efficacy,obs):
    # grant, for each modality, a rating (bet. 0 and 1) to account for how optimal the agent behaviour was overall.
    # A rating of 1 indicates that the agent pursued a behaviour that granted him with only maximum preference observations.
    Nmod = len(efficacy)
    evaluation = []
    for modality in range(Nmod):
        evaluation.append(0)

        No = efficacy[modality].shape[0]  #Number of outcomes for this modality
        counter = 0

        for t in range(tau):
            max_mark = np.max(efficacy[modality][:,t])

            observed = obs[modality,t]
            mark = efficacy[modality][observed,t]
            
            evaluation[modality] += (float(mark)/float(max_mark))
            #print(mark,max_mark,(float(mark)/float(max_mark)))
            counter += 1
        evaluation[modality] = evaluation[modality]/counter
    return evaluation