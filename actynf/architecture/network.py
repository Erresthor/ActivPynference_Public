from ..base.function_toolbox import isField
from ..layer.layer_link import get_layerLinks_between,str_layerLinks_between
import copy 
from .ordering_using_forces import mechanical_equations_ordering
from ..layer.model_layer import mdp_layer

import matplotlib.pyplot as plt
import numpy as np

class network():
    def __init__(self,in_layers=None,name=None,override_T = None,override_seed=None):
        if (not(isField(name))):
            self.name = "default_network"
        else :
            assert type(name)==str, "Network name should be a string, not " + str(name)
            self.name = name
        self.layers = []
        self.run_order=[]
        self.override_seed = override_seed

        if (isField(in_layers)):
            for lay in in_layers:
                assert (type(lay)==mdp_layer), "Error : " + str(lay) + " is of type " + str(type(lay)) + " instead of 'mdp_layer'"
                self.layers.append(lay)

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
        network_str += "   LINKS:\n"
        for lay in self.layers:
            for link in lay.get_links("from"):
                link_str = "    - " + str(link)
                link_str = '\t'.join(link_str.splitlines(True)) + "\n"
                network_str += link_str
        network_str += "___________________________________________________\n"
        return network_str
    
    def reseed(self,new_seed=None):
        if (isField(new_seed)):
            self.override_seed = new_seed
        if (isField(self.override_seed)):
            layer_cnt = 0
            for lay in self.layers:
                lay.reseed(self.override_seed+layer_cnt) # to avoid all layers with the same seed
                layer_cnt += 1

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

    def update_order_run(self,mean_of_last_percent = 0.25):
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
        list_of_neighbors = [] # per layer, the layers connected through upstream or downstream links
        for lay_idx in range(len(self.layers)) :
            lay = self.layers[lay_idx]
            linked_layers = lay.get_connection_weighted_linked_layers()

            # UPSTREAM LAYER -------------> CURRENT LAYER
            #                  layerLink
            # provides forces towards high x values for current
            upstream_neighbors = []
            for us in linked_layers["to_self"]:
                upstream_layer, cardinal, is_process_to_model = us
                if not(is_process_to_model):
                    upstream_layer_idx = self.layers.index(upstream_layer)
                    upstream_neighbors.append([upstream_layer_idx,cardinal])

            # CURRENT_LAYER -------------> DONWSTREAM LAYER
            #                  layerLink
            # provides forces towards low x values for current
            downstream_neighbors = []
            for ds in linked_layers["from_self"]:
                downstream_layer, cardinal, is_process_to_model = ds
                if not(is_process_to_model):
                    downstream_layer_idx = self.layers.index(downstream_layer)
                    downstream_neighbors.append([downstream_layer_idx,cardinal])
            list_of_neighbors.append([upstream_neighbors,downstream_neighbors])
        xs,xxs,xxxs = (mechanical_equations_ordering(list_of_neighbors,opt=1))

        self.run_order = list(np.argsort(np.mean(xs[:,int((1.0-mean_of_last_percent)*xs.shape[1]):],axis=1)))

    def prerun(self):
        for order_idx in self.run_order:
            self.layers[order_idx].prerun()
        
    def postrun(self):
        for order_idx in self.run_order:
            self.layers[order_idx].postrun()
        
    def run(self,verbose=True,return_STMs = False):
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
                searchtree = next(list_of_layer_tickgenerators[order_idx])
                updated_layer.transmit_outputs()
        if (verbose):
            print()
            seeds = [str(lay.seed)+"-"+str(lay.trials_with_this_seed) for lay in self.layers]
            print(" Done !   -------- (seeds : [" + ';'.join(seeds) + "])")
        self.postrun()

        if (return_STMs):
            # Indicators for all the layers :
            STMs_for_each_layer = []
            for lay in self.layers :
                # Store the STM at that point (before it is reinitialized)
                STMs_for_each_layer.append(lay.STM.copy())
            return STMs_for_each_layer

    def run_N_trials(self,N,small_verbose=True,big_verbose=False,return_STMs = False):
        STMlist = []
        for n in range(N):
            potential_STM_list = self.run(big_verbose,return_STMs) # None if return_STMs is False, otherwise a list of STMs for each layer
            if (return_STMs):
                STMlist.append(potential_STM_list)
        if (small_verbose):
            seeds = [str(lay.seed) for lay in self.layers]
            print(" Done !  -------- (seeds : [" + ';'.join(seeds) + "])")
        if (return_STMs):
            return STMlist
    
    def copy_network(self,copied_id,override_name=False,verbose=False):
        if (verbose):
            print("Copying network...",end="")
        copied_net = copy.deepcopy(self)

        if (override_name):
            copied_net.name = str(copied_id)
        else :
            copied_net.name = self.name + "_" + str(copied_id)
        
        for lay_idx in range(len(copied_net.layers)):
            lay = copied_net.layers[lay_idx]
            if (override_name):
                lay.name = str(lay_idx) + "_" + str(copied_id)
            else :
                lay.name = lay.name + "_" + str(copied_id)
        if (verbose):
            print(" Done !")
        return copied_net
        
        # All this may be useless as deepcopy seems to be
        # doing it much better :)
        # Create a dictionary to map original nodes to their copies
        copied_node_dict = {}
        copied_links_dict = {}

        starting_node = self.layers[0]
        # Perform a depth-first traversal of the network
        stack = [starting_node]
        while stack:
            current_node = stack.pop()
            print("   Copying layer " + current_node.name + "...")
            current_node_copy = current_node.copy()

            # Establish the connections in the copied network
            for original_node, copy_node in copied_node_dict.items():
                print("      - Checking links with " + original_node.name + "...")
                existing_connections = get_layerLinks_between(current_node,original_node)
                # layer_link_str = str_layerLinks_between(current_node,original_node)
                # print(layer_link_str)
                # print(existing_connections)
                for forward_link in existing_connections[0] :
                    copied_link = forward_link.copy(current_node_copy,copy_node,merge_verbose=merge_verbose)
                    copied_links_dict[forward_link] = copied_link
                for backward_link in existing_connections[1]:
                    copied_link = backward_link.copy(copy_node,current_node_copy,merge_verbose=merge_verbose)
                    copied_links_dict[backward_link] = copied_link
            copied_node_dict[current_node] = current_node_copy
        copied_network = network(list(copied_node_dict.values()),list(copied_links_dict.values()))
        print("Done !")
        return copied_network
        
        # copy_node.connections = [node_copies[conn] for conn in original_node.connections]


        #     # Copy the connections between input & ouptuts
        #     for input_connection in current_node.inputs.connections:
        #         # Check if the connection has been copied already
        #         if connection not in node_copies:
        #             stack.append(connection)
            
        #     copied_node_list.append(current_node_copy)

        # # Establish the connections in the copied network
        # for original_node, copy_node in copied_node_dict.items():
        #     copy_node.connections = [node_copies[conn] for conn in original_node.connections]

        # # Return the copy of the starting node
        # return node_copies[node]

    