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
from .active_model_container import ActiveModelSaveContainer
from .active_model_save_manager import ActiveSaveManager

class ActiveModel():
    def __init__(self,active_save_manager : ActiveSaveManager,modelname="default",savefolder=None):
        self.model_name = modelname
        self.save_folder = savefolder
        self.sample_size = 0

        self.A = None
        self.a = None

        self.B = None
        self.b = None

        self.C = None
        self.c = None

        self.D = None
        self.d = None

        self.E = None
        self.e = None

        self.U = None

        self.T = None

        self.s_ = None  # |
        self.o_ = None  # | Manual input for hidden states, observations and actions
        self.u_ = None  # |

        self.layer_options = mdp_layer_options()

        self.layer_options.T_horizon = 1
        self.layer_options.learn_during_experience = False
        self.layer_options.memory_decay = MemoryDecayType.NO_MEMORY_DECAY
        self.layer_options.decay_half_time = 500

        self.layer_list = []

        self.save_manager = active_save_manager
        self.save_manager.update_model_name(self.model_name)
        self.save_manager.update_folder_name(self.save_folder)
        self.saveticks = self.save_manager.list_of_intermediate_ticks()

    def initialize_layer(self):
        layer = mdp_layer()
        layer.Ni = 16

        layer.T = flexible_copy(self.T)

        layer.options = self.layer_options
        layer.options.T_horizon               = self.layer_options.T_horizon
        layer.options.learn_during_experience = self.layer_options.learn_during_experience
        layer.options.memory_decay            = self.layer_options.memory_decay
        layer.options.decay_half_time         = self.layer_options.decay_half_time

        layer.A_ = flexible_copy(self.A)
        layer.a_ = flexible_copy(self.a)
        
        layer.B_ = flexible_copy(self.B)
        layer.b_ = flexible_copy(self.b)

        layer.C_ = flexible_copy(self.C)
        layer.c_ = flexible_copy(self.c)

        layer.D_ = flexible_copy(self.D)
        layer.d_ = flexible_copy(self.d)
        
        layer.E_ = flexible_copy(self.E)
        layer.e_ = flexible_copy(self.e)

        layer.U_ = flexible_copy(self.U)

        layer.s_ = flexible_copy(self.s_)
        layer.o_ = flexible_copy(self.o_)
        layer.u_ = flexible_copy(self.u_)

        layer.gen = 0
        
        self.layer_list.append(layer)
        self.sample_size += 1
    
    def isModelSaved(self):
        model_config_path = os.path.join(self.save_folder,self.model_name,"_MODEL")
        return os.path.exists(model_config_path)
    
    def save_model(self,savelist = False):
        if (not(savelist)):
            temp_list = self.layer_list
            self.layer_list = []
        model_config_path = os.path.join(self.save_folder,self.model_name,"_MODEL")
        save_flexible(self,model_config_path)
        if (not(savelist)):
            self.layer_list = temp_list
    
    def load_model(model_folder_name):
        model_config_path = os.path.join(model_folder_name,"_MODEL")
        return (load_flexible(model_config_path))
    
    def initialize_n_layers(self,n):
        for k in range(n):
            self.initialize_layer()
            self.sample_size += 1
    
    def layer_generator(layer,state_transition_rule=None,obs_perception_rule=None,initial_state=None,initial_observation=None):
        """ Allows better control of a model run. The user can define initial states and observations, as well as complex transition and perception rules. 
        For now, pre-defined sequences of states, observations and actions using self.s_,self.u_ and self.o_ aren't implemented."""

        if layer==None :
            return False
        layer.prep_trial()
        

        next_real_state = np.full((layer.Nf,),-1) 
        if (isField(initial_state)):
            next_real_state = initial_state            

        next_observation = np.full((layer.Nmod,),-1) 
        if (isField(initial_observation)):
            next_observation = initial_observation

        for t in range(layer.T):
            layer.s[:,t] = next_real_state
            layer.o[:,t] = next_observation

            layer.tick()
            yield [layer,t]


            # Calculate real states and observations given our own observation rule
            # previous_real_states = index_to_dist(layer.s[:,t],initial_state)
            
            # New gt latent states are due to either manual formulation (state_transition_rule) or classic MDP_VB formulation (D-B)
            next_real_state = np.full(layer.s[...,t].shape,-1) 
            if (isField(state_transition_rule)): #if (isField(state_transition_rule)): # Manual formulation (if t = 0, we gave an initial state, else a state transition rule was defined)
                next_real_state = state_transition_rule(layer)

            # New observations are due to either manual formulation (obs_perception_rule) or classic hidden gt state-related MDP_VB formulation (A)
            next_observation = np.full(layer.o[...,t].shape,-1)
            if (isField(obs_perception_rule)): # Manual formulation (if t = 0, we gave an initial observation, else an observation perception rule was defined)
                next_observation = obs_perception_rule(layer)
        
        layer.postrun(True) # learning step
                     #return [layer,t]



    def run_trial(self,trial_counter,state_transition_rule=None,obs_perception_rule=None,initial_state=None,initial_observation=None,overwrite=False):
        """Initialize sample_size generators with the same rules.
            Possibility to introduce parrallel processing here ? """
        self.save_model()
        savebool = self.save_manager.save_this_trial(trial_counter)
        for k in range(len(self.layer_list)):
            print("(Model " + str(k)+ " )")
            lay = self.layer_list[k]

            # CHECK : does this trial already exist ?
            # Ask the save manager :
            
            run_next_trial = False 
            if(not(overwrite)):
                existbool = self.save_manager.check_exists(k,trial_counter,'f',lay)
                if (existbool):
                    print("Trial " + str(trial_counter) +" for instance " + str(k)+ " already exists.")
                else :
                    run_next_trial = True
            else :
                run_next_trial = True
            
            if run_next_trial :
                for ol in ActiveModel.layer_generator(lay,state_transition_rule,obs_perception_rule,initial_state,initial_observation):
                    t = ol[1]  # To get the actual timestep 
                    updated_layer = ol[0]
                    if ((t in self.saveticks)and(savebool)) :
                        print("----------------  SAVING  ----------------")
                        self.save_manager.save_process(updated_layer,trial_counter,k,t)
                if (savebool):
                    self.save_manager.save_process(updated_layer,trial_counter,k,'f')
                    # Save the trial AFTER the learning step !
                    
                print(" - Observations --> " + str(self.layer_list[k].o))
                print(" - Actual states --> " + str(self.layer_list[k].s))
                print(" - Belief about states --> \n" + str(np.round(self.layer_list[k].X,1)) + "\n")
                print(" - Chosen actions --> " + str(self.layer_list[k].u))
            
        # print(" - State perception --> " + str(self.layer_list[0].a_[0]))
        # for k in range(self.layer_list[0].b_[0].shape[-1]):
        #     print(" - Action perception --> " + str(self.layer_list[0].b_[0][:,:,k]))

    def run_n_trials(self,n,state_transition_rule=None,obs_perception_rule=None,initial_state=None,initial_observation=None,overwrite=False):
        for k in range(n):
            print("---------------")
            print("Trial " + str(k) + " .")
            print("---------------")
            self.run_trial(k,state_transition_rule,obs_perception_rule,initial_state,initial_observation,overwrite=overwrite)
