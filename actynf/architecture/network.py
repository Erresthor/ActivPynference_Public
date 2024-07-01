
import matplotlib.pyplot as plt
import numpy as np
import copy

from ..base.function_toolbox import isField 
from .ordering_using_forces import mechanical_equations_ordering
from ..layer.model_layer import mdp_layer,layerMode


class network():
    def __init__(self,in_layers=None,name=None,override_T = None,override_seed=None):
        if (not(isField(name))):
            self.name = "default_network"
        else :
            assert type(name)==str, "Network name should be a string, not " + str(name)
            self.name = name
        self.layers = []
        self.run_order= []
        self.override_seed = override_seed

        if (isField(in_layers)):
            for lay in in_layers:
                assert (type(lay)==mdp_layer), "Error : " + str(lay) + " is of type " + str(type(lay)) + " instead of 'mdp_layer'"
                self.layers.append(lay)
            self.update_order_run()

        self.T = None
        if (isField(override_T)):
            self.T = override_T
        else : 
            self.update_T()

        self.reseed() # If an override_seed was provided, the network object will change 
                      # its layers seeds !
            
    def __str__(self):
        network_str = "___________________________________________________\n"
        network_str += "LAYER NETWORK " + self.name + " : \n"
        network_str += "___________________________________________________\n"
        network_str += "   LAYERS:\n"
        for lay in self.layers:
            network_str += "    - " + str(lay.name) + "\n"
        network_str += "___________________________________________________\n"
        return network_str
    
    def reseed(self,new_seed=None,layers_auto_reseed=False):
        if (isField(new_seed)):
            self.override_seed = new_seed

        if (isField(self.override_seed)):
            layer_cnt = 0
            for lay in self.layers:
                lay.reseed(self.override_seed+layer_cnt) # to avoid all layers with the same seed
                layer_cnt += 1
        else :
            if layers_auto_reseed :
                for lay in self.layers:
                    lay.reseed(auto_reseed=True) # to avoid all layers running with the same seed

    def update_T(self):
        """ Using the layers of the network, guess how many timesteps per trial the network should run."""
        layers_T_list = []
        for lay in self.layers:
            layers_T_list.append(lay.T)
        if (len(layers_T_list) >0):
            self.T = max(layers_T_list)
            return True
        else : 
            return False

    def update_layer_dependencies(self,update_dependents=True):
        for lay in self.layers:
            lay.update_sources(update_dependents)
    
    def update_order_run(self,mean_of_last_percent = 0.25,show_plot=False):
        """
        General philosophy : inputs & outputs should NOT be cleaned after a run : some layers 
        will use the same input over two trials dependind on the general ordering of the fire events. 
        A layerLink is a directional object that sends data from a layer to another.
        By iterating over layerlinks, we may build a general gradient of layer run orders : 
        ideally, this should depend on the "strength" of the layerLinks
        """
        assert len(self.layers) > 0, "There are no layers in the network " + self.name + " . Please add layers to the network before running network.update_order_run ."
        # 1. Get a list of upstream and downstream layers for each layer. 
        # If a dowstream layer is in generative process mode and the source
        # layer is in generative model mode, ignore the downstream layer.
        K = 1
        self.update_layer_dependencies(True)

        list_of_neighbors = [] # per layer, the layers connected through upstream or downstream links
        
        
        for lay_idx in range(len(self.layers)) :
            lay = self.layers[lay_idx]
            
            # UPSTREAM LAYER -------------> CURRENT LAYER
            # provides forces towards high x values for current
            upstream_neighbors = []
            for source_lay in lay.sources:
                # If my source is a model, and i am a process, let's ignore that !
                if not((source_lay.layerMode == layerMode.MODEL) and (lay.layerMode == layerMode.PROCESS)):
                    upstream_neighbors.append([self.layers.index(source_lay),K])
            
            # CURRENT_LAYER -------------> DOWNSTREAM LAYER
            # provides forces towards low x values for current
            downstream_neighbors = []
            for dependent_lay in lay.dependent:
                # If i am a process, let's ignore that !
                if not((dependent_lay.layerMode == layerMode.PROCESS) and (lay.layerMode == layerMode.MODEL)):
                    downstream_neighbors.append([self.layers.index(dependent_lay),K])

            list_of_neighbors.append([upstream_neighbors,downstream_neighbors])
        xs,xxs,xxxs = (mechanical_equations_ordering(list_of_neighbors,opt=1))
        if show_plot:
            # Show the output of mechanical ordering
            for i in range(len(self.layers)):
                plt.plot(np.linspace(0,xs.shape[-1],xs.shape[-1]),xs[i,:])
                plt.xlabel("Iterations")
            plt.show()
        self.run_order = list(np.argsort(np.mean(xs[...,int((1.0-mean_of_last_percent)*xs.shape[1]):],axis=1)))
    
    def prerun(self):
        for order_idx in self.run_order:
            self.layers[order_idx].prerun()
        
    def postrun(self):
        for order_idx in self.run_order:
            self.layers[order_idx].postrun()
    
    def get_current_layers_weights(self):
        weights_for_each_layer = []
        for lay in self.layers :
            # Store the layer weights at that point (before it is reinitialized)
            weights_for_each_layer.append(lay.get_weights())
        return weights_for_each_layer
    
    def get_current_layers_stms(self):
        # Indicators for all the layers :
        STMs_for_each_layer = []
        for lay in self.layers :
            # Store the STM at that point (before it is reinitialized)
            STMs_for_each_layer.append(lay.STM.copy())
        return STMs_for_each_layer
    
    def run(self,verbose=True,
            return_STMs = True,return_weights = True    ):
        assert len(self.layers) > 0, "There are no layers in the network " + self.name + " . Please add layers to the network before running network.run ."
        
        if (len(self.run_order)==0):
            self.update_order_run()
        if (not(isField(self.T))):
            if (not(self.update_T())):
                raise ValueError("Could not find a suitable value for T. Aborting run ...")
            
        self.prerun()
        list_of_layer_tickgenerators = ([layr.tick_generator() for layr in self.layers])
        for timestep in range(self.T):
            if (verbose) : 
                print(" Network [" + self.name + "] : Timestep " + str(timestep+1) + " / " + str(self.T), end='\r')
            for order_idx in self.run_order:
                updated_layer = self.layers[order_idx]
                try:
                    searchtree = next(list_of_layer_tickgenerators[order_idx])
                except :
                    raise RuntimeError("ERROR in network <"+ self.name + "> - layer [" + (self.layers[order_idx]).name + "] :")
                # updated_layer.transmit_outputs()
        if (verbose):
            print()
            seeds = [str(lay.seed)+"-"+str(lay.trials_with_this_seed) for lay in self.layers]
            print(" Done !   -------- (seeds : [" + ';'.join(seeds) + "])")
        self.postrun()

        
        weights_for_each_layer = None
        if return_weights:
            weights_for_each_layer = self.get_current_layers_weights()

        STMs_for_each_layer = None
        if return_STMs:
            STMs_for_each_layer = self.get_current_layers_stms()
        
        return STMs_for_each_layer,weights_for_each_layer

    def run_N_trials(self,N,small_verbose=True,big_verbose=True,
                     return_STMs = True,return_weights = True):
        STMlist = []
        weightlist = []
        if (return_weights):
            # We gather the initial weights of the layer at index 0
            weightlist.append(self.get_current_layers_weights())
        if (return_STMs):
            STMlist.append(None) # To match the indices
            
        for trial_id in range(N):
            potential_STM_list,potential_weightlist = self.run(big_verbose,return_STMs=return_STMs,return_weights=return_weights) # None if return_STMs is False, otherwise a list of STMs for each layer
            if (return_STMs):
                STMlist.append(potential_STM_list)
            if (return_weights):
                weightlist.append(potential_weightlist)
        if (small_verbose):
            seeds = [str(lay.seed) for lay in self.layers]
            print(" Done !  -------- (seeds : [" + ';'.join(seeds) + "])")

        return (STMlist if return_STMs else None),(weightlist if return_weights else None)
    
    def copy_network(self,copied_id,override_name=False,verbose=False,
                    new_net_seed=None,same_seed=False):
        if (verbose):
            print("Copying network...",end="")
        copied_net = copy.deepcopy(self)

        if (override_name):
            copied_net.name = str(copied_id)
        else :
            copied_net.name = self.name + "_" + str(copied_id)
        
        if isField(new_net_seed):
            assert type(self.seed)==int,"Seed should be an integer ..."
            self.override_seed = new_net_seed
            self.reseed()
        elif (not(same_seed)):
            self.reseed(layers_auto_reseed=True)


        for lay_idx in range(len(copied_net.layers)):
            lay = copied_net.layers[lay_idx]
            if (override_name):
                lay.name = str(lay_idx) + "_" + str(copied_id)
            else :
                lay.name = lay.name + "_" + str(copied_id)
        if (verbose):
            print(" Done !")
        return copied_net