import os,sys
import string
from pyexpat import model
from enum import Enum
import random as r
import time 
import numpy as np
import matplotlib.pyplot as plt
from scipy import rand

from ..base.miscellaneous_toolbox import isField,flexible_copy
from ..base.file_toolbox import load_flexible,save_flexible
from.active_model_container import ActiveModelSaveContainer
#from pynf_functions import *


class ActiveSaveManager():
    def __init__(self,T,trial_savepattern=1,intermediate_savepattern=0,modelname="default",folder_name=None):
        self.model_name = modelname
        self.folder_name = folder_name
        self.trial_savepattern = trial_savepattern
        self.intermediate_save_pattern = intermediate_savepattern
        self.T = T

    def update_model_name(self,model_name):
        self.model_name = model_name
    
    def update_folder_name(self,folder_name):
        self.folder_name = folder_name
    
    def generate_save_name(model_folder,instance,trial,timestep):
        if not(type(timestep)==str):
            t_counter_string = f'{timestep:07d}'
        else : 
            t_counter_string = timestep
        parrallel_counter_string = f'{trial:09d}'
        
        instance_string = f'{instance:03d}'
        name = os.path.join(model_folder,instance_string,parrallel_counter_string + "_" + t_counter_string)
        return name
    
    def save_process(self,layer,trial_counter,parrallel_counter,t):
        assert isField(self.folder_name),"Please provide a folder name to the active save manager."
        model_folder = os.path.join(self.folder_name,self.model_name)
        
        wholename = ActiveSaveManager.generate_save_name(model_folder,parrallel_counter,trial_counter,t)
        
        container = ActiveModelSaveContainer(wholename,layer,trial_counter)
        container.layer_instance = parrallel_counter
        print("Saving to : " + container.path)
        container.quicksave()

    def save_this_instance(self,instance):
        assert type(instance)==int, "Instance should be an integer ..."
        if (instance==0):
            return True
        return (instance%self.trial_savepattern==0)

    def list_of_intermediate_ticks(self):
        """Depending on the savepattern, the saved ticks is the list of ticks the save manager will save in each run. Example : if a  trial is made of 5 tmstps, a possible output would be [0,2,4]"""

        if (self.T == 0):
            return

        sp =self.intermediate_save_pattern
        savepattertype = type(self.intermediate_save_pattern)

        if (savepattertype==list):
            for k in range(len(sp)):
                assert (sp[k]<=self.T), "If a list is given as a save pattern, its elements should be inferior to T = " + str(self.T)
            return self.save_pattern
        elif (savepattertype==float):
            assert (sp>=0)and(sp<=1),  "If a ratio is given as a save pattern, it should stand between 0 and 1"
            
            if(sp == 0):
                return [self.T]

            savelist= [self.T] 
            current_ratio = 1.0/self.T

            # We want       current ratio <= sp < current_ratio + 1/T
            
            limiter = 2*self.T
            cnt = 0
            while ((current_ratio < sp)and(cnt<limiter)) :
                k = int(r.random()*(self.T))
                if (not(k in savelist)):
                    savelist.append(k)
                current_ratio = len(savelist)/self.T
                cnt += 1
            return savelist
        elif (savepattertype==int):
            assert (sp<=self.T),  "If a count is given as a save pattern, it should stand between 0 and T = " + str(self.T)
            
            if(sp == 0):
                return []

            savelist= [self.T - 1] 
            current_counter = 1.0

            # We want       current ratio <= sp < current_ratio + 1/T
            
            limiter = 5*sp
            cnt = 0
            while ((current_counter < sp)and(cnt<limiter)) :
                k = int(r.random()*(self.T-1))
                if (not(k in savelist)):
                    savelist.append(k)
                current_counter += 1
                cnt += 1
            return savelist
        else :
            print("Save pattern type ( " + str(savepattertype) + " )  not recognized ... Aborting.")

    def get_results_loader(self):
        
        model_folder = os.path.join(self.folder_name,self.model_name) 

        def loader(model_instance,trial,timestep):
            try :
                path = ActiveSaveManager.generate_save_name(model_folder,model_instance,trial,timestep)
                container = ActiveModelSaveContainer.load_active_model_container(path)
                return container
            except : 
                return None

        return loader