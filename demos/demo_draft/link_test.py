import numpy as np

from actynf.base.function_toolbox import normalize
from actynf.layer.model_layer import layer_output,mdp_layer,layer_input
from actynf.layer.layer_link import layerLink,get_layerLinks_between,establish_layerLink
from actynf.architecture.network import network

class tester :
    def __init__(self):
        self.basic_arr = np.array([[1,0,0,0],
                      [0,1,0,0]])

    def return_part_of_basicarr(self,idx):
        return self.basic_arr[idx,:]

def test_this(k):
    mytest = tester()
    getattr(mytest,"basic_arr")[k,:] = np.array([-256,-256,-256,-256])
    print(mytest.basic_arr)

def linktest():
    a = [np.random.random((4,3,1)),np.random.random((2,3,1))]
    b = normalize([np.random.random((3,3,2)),np.random.random((1,1,3))])
    c = [np.array([0,0,1,0]),np.array([1,1])]
    d = normalize([np.array([1,0,0]),np.array([1])])
    e = np.array([1,1,1])
    u = np.array([[0,1],[0,2],[1,0]])
    T = 100
    model_layer_test = mdp_layer("generative_process",a,b,c,d,e,u,T)
    # model_layer_test.primeModelFunctions()

    a = [np.ones((1,2,4)),np.ones((3,4,4))]
    b = [np.ones((2,2,2)),np.ones((4,4,3))]
    c = [np.random.random((1,)),np.random.random((3,))]
    d = [np.random.random((2,)),np.random.random((4,))]
    e = np.array([1,1,1])
    u = np.array([[0,1],[0,2],[1,0]])
    layer_2 = mdp_layer("generative_model",a,b,c,d,e,u,T)

    outo = model_layer_test.outputs
    into = layer_2.inputs

    linker = layerLink(outo, into,[["o.1-0","s"]])
    # linker2 = layerLink(into, outo,[["o.1","s_d.0"]])
    # model_layer_test.outputs.o_d = normalize(np.random.random((4,2)),all_axes=True)
    # linker.fire_all_connections()

    layer_2_copy = layer_2.copy()
    model_layer_test_copy = model_layer_test.copy()
    layer_2_copy = layer_2.copy()
    new_linker = linker.copy(model_layer_test_copy,layer_2_copy)
    new_linker_3 = linker.copy(model_layer_test_copy,layer_2_copy)
    new_linker_5 = linker.copy(model_layer_test_copy,layer_2_copy)
    new_linker_7 = linker.copy(model_layer_test_copy,layer_2_copy)
    # print(model_layer_test_copy.outputs.connections)
    K = get_layerLinks_between(model_layer_test_copy,layer_2_copy)
    print(K)

def network_test():
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

    linker_obs = establish_layerLink(layer_1, layer_2,[["o.1","o.0"]])
    linker_obs2 = establish_layerLink(layer_1, layer_2,["o.0","o.1"])
    linker_act = establish_layerLink(layer_2, layer_1,[["u","u"]])


    NNetCopies = 10
    NTrials = 10

    net = network([layer_1,layer_2],"bruh_network")
    # net = network()
    nets = []   
    for k in range(NNetCopies):
        nets.append(net.copy_network(k,verbose=True))

    # for layer in nets[0].layers :
    #     print(layer.get_connection_weighted_linked_layers())
    # nets[0].update_order_run()
    for trial in range(NTrials):       
        for net in nets :
            net.run()

    for net in nets :
        print()
        print(np.round(normalize(net.layers[1].a[0]),2))