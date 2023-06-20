import numpy as np
from pyai.base.function_toolbox import normalize
from pyai.layer.model_layer import layer_output,mdp_layer,layer_input
from pyai.architecture.layer_link import layerLink

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
    # layer_2.primeModelFunctions()

    outo = model_layer_test.outputs
    into = layer_2.inputs

    linker = layerLink(outo, into,[["o.0","s.1"],["o.1","s.0"]])
    linker2 = layerLink(into, outo,[["o.1","s_d.0"]])

    linker.fire_all_connections()
    print(linker.connections)
    print(linker2.connections)