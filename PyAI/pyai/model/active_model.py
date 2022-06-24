#!/bin/python
# -*- coding: utf-8 -*-

"""
Created on Tue Aug 3 10:55:21 2021

@author: cjsan
"""
from subprocess import list2cmdline
from matplotlib.cbook import maxdict
import numpy as np
import random
from PIL import Image, ImageDraw
import sys,os,inspect
import math
import statistics
import random as r
import os.path
from scipy import stats
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time

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
    def __init__(self,active_save_manager : ActiveSaveManager = None,modelname=None,savefolder=None,verbose = False):
        self.index = None # This is a tuple/list that we use to preserve parameter coherence
                          # Used to build cross-model representations
        self.input_parameters = None

        self.verbose = verbose
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

        if (isField(active_save_manager)):
            self.save_manager = active_save_manager
            self.model_name = active_save_manager.model_name
            self.save_folder = active_save_manager.folder_name
        else :
            if (isField(modelname)):
                self.model_name = modelname
            if(isField(savefolder)):
                self.save_folder = savefolder
        
        self.saveticks = self.save_manager.list_of_intermediate_ticks()

        self.custom_state_transition_rule=None
        self.initial_state=None

        self.custom_obs_perception_rule=None
        self.initial_observation=None

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

    def run_trial(self,trial_counter,state_transition_rule=None,obs_perception_rule=None,initial_state=None,initial_observation=None,overwrite=False,global_prop=None,list_of_last_n_trial_times=None):
        """Initialize sample_size generators with the same rules.
            Possibility to introduce parrallel processing here ? """
        
        savebool = (self.save_manager.save_this_trial(trial_counter))and(isField(self.save_manager))

        if (savebool)and((trial_counter==0)or(not(self.isModelSaved()))):
            self.save_model()

        for k in range(len(self.layer_list)):
            # ---------------------- THIS IS JUST COSMETIC ---------------------------------
            if(self.verbose):
                print("(Model " + str(k)+ " )")

                
                if (not(global_prop==None)) :
                    model_n = global_prop[0]
                    max_model_n = global_prop[1]
                    trial_n = global_prop[2]
                    max_trial_n = global_prop[3]
                    instance_n = k # local instance index
                    max_instance_n = len(self.layer_list)

                    max_total_instances = max_model_n*max_trial_n*max_instance_n

                    current_total_model_counter = model_n
                    #invert instance and trial enumeration due to strange active model organization
                    current_total_trial_counter = max_trial_n*current_total_model_counter + trial_n
                    current_total_instance_counter = max_instance_n*current_total_trial_counter + instance_n
                    #current_total_trial_counter = max_trial_n*current_total_instance_counter + trial_n
                    
                    total_progress = int(1000*(100*float(current_total_instance_counter)/max_total_instances))/1000.0
                    #print("Model number : " + str(current_total_model_counter))
                    print("----  " + str(total_progress) + "  % ----")
                    if (not(list_of_last_n_trial_times==None)):
                        mean_time = statistics.mean(list_of_last_n_trial_times)
                        print("Mean time for a trial : " + str(mean_time))
                        total_time = mean_time*max_total_instances
                        done_time = mean_time*current_total_instance_counter
                        remaining_time = mean_time*(max_total_instances-current_total_instance_counter)


                        days = remaining_time//(3600*24)
                        hours = (remaining_time - days*3600*24)//3600
                        mins = (remaining_time - days*3600*24 - hours*3600)//60
                        secs = (remaining_time -60*(mins + 60*(hours + 24*(days))))
                        print("Remaining time :    " + str(int(days)) + " Days - "+str(int(hours))+" Hours - " +str(int(mins)) +" Min - "+str(int(secs))+" Sec")
                    print("------" + "-----" + "--------")
            # ---------------------- THIS IS JUST COSMETIC ---------------------------------
            
            
            lay = self.layer_list[k]

            # CHECK : does this trial already exist ?
            # Ask the save manager :
            if(not(overwrite))and(savebool):
                run_next_trial = False 
                existbool = self.save_manager.check_exists(k,trial_counter,'f',lay)
                if (existbool)and(self.verbose):
                    print("Trial " + str(trial_counter) +" for instance " + str(k)+ " already exists.")
                else :
                    run_next_trial = True
            else :
                run_next_trial = True
            
            tbefore = time.time()
            if run_next_trial :
                for ol in ActiveModel.layer_generator(lay,state_transition_rule,obs_perception_rule,initial_state,initial_observation):
                    t = ol[1]  # To get the actual timestep 
                    updated_layer = ol[0]
                    if ((t in self.saveticks)and(savebool)) :
                        if(self.verbose):
                            print("----------------  SAVING  ----------------")
                        self.save_manager.save_process(updated_layer,trial_counter,k,t)
                if (savebool):
                    self.save_manager.save_process(updated_layer,trial_counter,k,'f')
                    # Save the trial AFTER the learning step !
                
                if(self.verbose):
                    print("----------------  SAVING  ----------------")
                    print(" - Observations --> " + str(self.layer_list[k].o))
                    print(" - Actual states --> " + str(self.layer_list[k].s))
                    print(" - Belief about states --> \n" + str(np.round(self.layer_list[k].X,1)) + "\n")
                    print(" - Chosen actions --> " + str(self.layer_list[k].u))
            
                tafter = time.time()
            else : 
                tafter = tbefore + 0.001
            
            if (not(list_of_last_n_trial_times==None)):
                ntrialtimes = 100000
                list_of_last_n_trial_times.append(tafter-tbefore)
                if(len(list_of_last_n_trial_times)>ntrialtimes):
                    list_of_last_n_trial_times.pop(0)

    def run_n_trials(self,n,state_transition_rule=None,obs_perception_rule=None,initial_state=None,initial_observation=None,overwrite=False,global_prop=None,list_of_last_n_trial_times = None):
        # global_prop is the proportion of the overall number of models ran
        # aka if global prop is 0.5, we already did 50% of all models
        
        for k in range(n):
            if(self.verbose):
                print("---------------")
                print("Trial " + str(k) + " .")
                print("---------------")
            if (not(global_prop==None)):
                model_n = global_prop[0]
                max_model_n = global_prop[1]
                global_prop = [model_n,max_model_n,trial_n,max_trial_n]
            trial_n = k
            max_trial_n = n
            self.run_trial(k,state_transition_rule,obs_perception_rule,initial_state,initial_observation,overwrite=overwrite,global_prop=global_prop,list_of_last_n_trial_times=list_of_last_n_trial_times)
