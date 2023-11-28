from actynf.architecture.network import network
from actynf.layer.model_layer import mdp_layer
from actynf.layer.layer_components import link_function

from actynf.base.function_toolbox import normalize

import numpy as np
import random 

def network_test(NNetCopies = 10):
    a = [np.random.random((4,3,1)),np.random.random((2,3,1))]
    b = normalize([np.random.random((3,3,2)),np.random.random((1,1,3))])
    c = [np.array([0,0,1,0]),np.array([1,1])]
    d = normalize([np.array([1,0,0]),np.array([1])])
    e = np.array([1,1,1])
    u = np.array([[0,1],[0,2],[1,0]])
    T = 10
    layer_1 = mdp_layer("generative_process","process",a,b,c,d,e,u,T)
    # model_layer_test.primeModelFunctions()

    a = [np.ones((2,2,4)),np.ones((4,2,4))]
    b = [np.random.random((2,2,2)),np.random.random((4,4,3))]
    c = [np.random.random((2,)),np.random.random((4,))]
    d = [np.random.random((2,)),np.random.random((4,))]
    e = np.array([1,1,1])
    u = np.array([[0,1],[0,2],[1,0]])
    layer_2 = mdp_layer("generative_model","model",a,b,c,d,e,u,T)
    layer_2.hyperparams.alpha = 1

    def observation_pipeline(lay1):
        # Function that gives a layer's observation output
        subject_observations = np.array([lay1.o[1],lay1.o[0]])
        return subject_observations
    layer_2.inputs.o = link_function(layer_1, observation_pipeline)

    action_pipeline = (lambda lay2:lay2.u) # Function that gives a layer's action output
    layer_1.inputs.u = link_function(layer_2,action_pipeline)


    net = network([layer_1,layer_2],"bruh_network")

    print(net.run_order)
    print(type(net.layers[1].inputs.o))
    net.run()


    # # net = network()
    nets = []
    for k in range(NNetCopies):
        nets.append(net.copy_network(k,verbose=True))
        nets[-1].reseed(random.randint(0,9999))
    net.run()

    # # for layer in nets[0].layers :
    # #     print(layer.get_connection_weighted_linked_layers())
    # # nets[0].update_order_run()
    # for trial in range(NTrials):
    for net in nets :
        # Check that all those objects are different !
        # print(id(net.layers[1].inputs.o.from_layers[0]))

        
        net.run()
        print(net.layers[1].STM.u_d)

    # for net in nets :
    #     print()

if __name__ == '__main__':
    network_test()