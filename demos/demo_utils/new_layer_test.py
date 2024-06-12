import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from actynf.layer.model_layer import mdp_layer
from actynf.layer.layer_learn import learn_from_experience

from actynf.base.function_toolbox import normalize,spm_complete_margin,spm_kron
from actynf.base.function_toolbox import sample_distribution
import actynf.old.layer_old.mdp_layer as old_lay


def test_old_layer(a,b,c,d,u):
    mdp_old = old_lay.mdp_layer()
    mdp_old.T = 10
    mdp_old.options.T_horizon = 2
    mdp_old.options.learn_during_experience = True
    mdp_old.Ni = 16
    mdp_old.A_ = [a]
    mdp_old.a_ = [a]
    mdp_old.D_ = d
    mdp_old.d_ = d
    mdp_old.B_  = b
    mdp_old.b_ = b
    C = np.array([[0],[0],[1],[0]])
    mdp_old.C_ = [C]
    mdp_old.U_ = u
    model_layer_test.belief_propagation()
    mdp_old.prep_trial()
    mdp_old.o[0,0] = model_layer_test.o[0,0]
    # mdp_old.tick()

def dist_from_definite_outcome(outcomeArray,No):
    return_dist_list = []
    for k in range(outcomeArray.shape[0]):
        return_dist_list.append(np.zeros((No[k],)))
        return_dist_list[k][outcomeArray[k]] = 1
    return_dist = np.zeros(tuple(No))
    return_dist[tuple(outcomeArray)]= 1
    return return_dist,return_dist_list

def test_generator():
    k = 0
    while k < 100:
        yield k
        k = k+1

def run_test():
    a = [np.random.random((4,3,2)),np.random.random((2,3,2))]
    b = normalize([np.random.random((3,3,2)),np.random.random((2,2,3))])
    c = [np.array([0,0,1,0]),np.array([1,1])]
    d = normalize([np.array([1,0,0]),np.array([1,1])])
    e = np.array([1,1,1])
    # a = np.ones((4,3,2))
    # b = [np.ones((3,3,2)),np.ones((2,2,3))]
    # c = np.array([0,0,1,0])
    # d = [np.array([1,0,0]),np.array([1,1])]
    # e = np.array([1,1,1])
    # u = np.array([[0,1],[0,2],[1,0]])
    u = np.array([[0,1],[0,2],[1,0]])
    T = 100
    model_layer_test = mdp_layer(a,b,c,d,e,u,T)
    model_layer_test.name = "my_layer"
    model_layer_test.primeModelFunctions()
    print(model_layer_test)
    

    # Test belief update
    all_obs_from_this = normalize(np.random.random((4,2)),all_axes=True)
    model_layer_test.inputs.o = np.array([2,0])
    for t in range(model_layer_test.T):
        obss = sample_distribution(all_obs_from_this)
        # model_layer_test.inputs.o_d = all_obs_from_this
        model_layer_test.inputs.o = np.asarray(obss)
        print(model_layer_test.inputs)
        model_layer_test.model_tick()
        # print(model_layer_test.outputs)
    for k in range(200):
        model_layer_test.learn()
    # learn_from_experience(model_layer_test)
    print(model_layer_test)
    # # Test process update :
    # force_this_observation_dist = np.array([[0.5,0.25],
    #                                         [0.0,0.15],
    #                                         [0.09,0.01],
    #                                         [0,0]])
    # model_layer_test.inputs.s = np.array([0,0])
    # model_layer_test.inputs.u = np.array([1])
    # model_layer_test.inputs.o_d = force_this_observation_dist
    # for k in range(T):
    #     print(k)
    #     print(model_layer_test.inputs)
    #     model_layer_test.process_tick()
    #     print(model_layer_test.inputs)
        