#!/bin/python
# -*- coding: utf-8 -*-

"""
Created on Tue Aug 3 10:55:21 2021

@author: cjsan
"""
import numpy as np
import random
from PIL import Image, ImageDraw
import sys,os,inspect
import math
import random as r
import os.path
from scipy import stats
import plotly.graph_objects as go
import matplotlib.pyplot as plt


from ..base.miscellaneous_toolbox import flexible_copy , isField , index_to_dist, dist_to_index
from ..base.function_toolbox import normalize
from ..base.plotting_toolbox import multi_matrix_plot
from ..base.file_toolbox import load_flexible,save_flexible
from ..base.function_toolbox import spm_dot,spm_kron
from ..base.matrix_functions import *
from ..base.file_toolbox import root_path
from ..layer.mdp_layer import mdp_layer,mdp_layer_options
from ..layer.layer_postrun import evaluate_run
from ..layer.layer_learn import MemoryDecayType
from ..layer.layer_sumup import *

from.active_model import ActiveModel
from .active_model_container import ActiveModelSaveContainer
from .active_model_save_manager import ActiveSaveManager


def show_figures(active_model,save_container,realA_override=None,realB_override=None,realD_override=None):
    t = save_container.t
    
    Nfactor = len(active_model.d)
    Nmod = len(active_model.a)
    
    for f in range(Nfactor):
        prior_d = active_model.d[f]
        prior_b = active_model.b[f]

        current_d = save_container.d_[f]
        current_b = save_container.b_[f]
        
        if not(isField(realB_override)) :
            real_B = active_model.B[f]
        else : 
            real_B = realB_override[f]
        
        if not(isField(realD_override)) :
            real_D = active_model.D[f]
        else : 
            real_D = realD_override[f]
    
    for mod in range(Nmod): 
        prior_a = active_model.a[mod]
        current_a = save_container.a_[mod]
        if not(isField(realA_override)) :
            real_A = active_model.A[mod]
        else : 
            real_A = realA_override[mod]
    
    multi_matrix_plot([normalize(real_B),normalize(prior_b),normalize(current_b)],["Real B","Prior b at t=0 ","Learnt b at t=" +str(t) + " "],"FROM states","TO states")
    multi_matrix_plot([normalize(real_A),normalize(prior_a),normalize(current_a)], ["Real A","Prior a at t=0 ","Learnt a at t=" +str(t) + " "],"State (cause)","Observation (consequence)")
    multi_matrix_plot([normalize(real_D),normalize(prior_d),normalize(current_d)], ["Real D","Prior d at t=0 ","Learnt d at t=" +str(t) + " "], "Initial belief","State")
    plt.show(block=False)
    input()
    plt.close()


def open_model_container(model_folder,model_instance_indice,trial,timestep):
    complete_name = ActiveSaveManager.generate_save_name(model_folder,model_instance_indice,trial,timestep)
    container = ActiveModelSaveContainer.load_active_model_container(complete_name)
    return container

def load_containers_in_folder(folder_name):
    container_list = []
    for file in os.listdir(folder_name):
        filepath = os.path.join(folder_name,file)
        container_list.append( ActiveModelSaveContainer.load_active_model_container(filepath))
    return container_list


def evaluate_trajectory_against_optimal(trajectory,optimal,metric = 'linear',mean=True):
    """A distance evaluation for hidden state trajectories of type np.array([1,2,3,1 1,0,0]) """
    assert trajectory.shape == optimal.shape , "Trajectory and optimal should possess equal dimensions."

    distance = 0
    
    if (metric=='linear'):
        returner = np.sum(np.abs(trajectory-optimal))
    elif(metric=='binary'):
        returner = np.sum(np.abs(trajectory!=optimal))
    elif(metric=='quadratic') :
        d2 = np.sum((trajectory-optimal)*(trajectory-optimal))
        returner = np.sqrt(d2)
    elif(metric=='infinity') :
        d2 = np.max(np.abs(trajectory-optimal))
        returner = np.sqrt(d2)
    else :
        for idx, x in np.ndenumerate(trajectory):
            print(idx)
        returner = 0
    if (mean):
        return (returner / trajectory.size)
    else :
        return returner
