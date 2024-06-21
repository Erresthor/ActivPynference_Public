import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import sys,os,inspect
import math
from scipy.interpolate import splprep,splev

from actynf.base.miscellaneous_toolbox import isField
from actynf.base.function_toolbox import normalize

from actynf.layer.model_layer import mdp_layer
from actynf.layer.layer_link import establish_layerLink
from actynf.architecture.network import network

def time_of_day_perceptron(T,Th,seed=None,isModel=False):
    No = [3,3] # 3 levels of luminosity, 3 levels of temperature
    Ns = 4 # We consider a day cut in 4 times : morning, day, evening and night

    d = [np.zeros((Ns,))]

    p_transition = 1.0
    b = [np.array([
        [1-p_transition,0.00,0.00,p_transition],
        [p_transition,1-p_transition,0.00,0.00],
        [0.00,p_transition,1-p_transition,0.00],
        [0.00,0.00,p_transition,1-p_transition]
    ])]
    b[0] = np.expand_dims(b[0],axis=-1)

    # lumi = f(time_of_day)
    a1 = np.array([
        [0.25,0.75,0.25,0.00],
        [0.50,0.25,0.50,0.25],
        [0.25,0.00,0.25,0.75]
    ])

    # temperature = f(time_of_day)
    a2 = np.array([
        [0.15,0.55,0.15,0.20],
        [0.70,0.25,0.70,0.25],
        [0.15,0.20,0.15,0.55]
    ])

    a = [a1,a2]

    c = [np.array([0,0,0]),np.array([0,0,0])] # We have no intrinsic preferences regarding temperature & luminosity

    u = np.array([0])
    e = np.ones(u.shape)

    if isModel:
        tod_perceptron = mdp_layer("time_of_day_perception_model","model",a,b,c,d,e,u,T,Th,in_seed=seed)
        tod_perceptron.learn_options.learn_a = False
        tod_perceptron.learn_options.learn_b = False
        tod_perceptron.learn_options.learn_d = False
        tod_perceptron.hyperparams.cap_action_explo = 2
        tod_perceptron.hyperparams.cap_state_explo = 1
        tod_perceptron.hyperparams.alpha = 10
    else :
        tod_perceptron = mdp_layer("time_of_day_perception_process","process",a,b,c,d,e,u,T,Th,in_seed=seed)
    return tod_perceptron

def hide_seek_layer(T,Th,seed=None):
    No = [4,3,2] # 4 levels of time of day, 3 levels of observed objects
    Ns = [2,2] # The agent is either inside or outside

    # State 1 : outside is dangerous / outside is safe
    # State 2 : i am inside / i am outside
    
    # Observation 1 : what time of day is it ?
    # Observation 2 : i find food / i find nothing / i find a predator !
    # Observation 3 : am I inside or outside ?

    # tod = f(danger,refuge), according to our agent, the time of day is a consequence of the "danger" hidden state
    a1 = np.zeros((No[0],Ns[0],Ns[1]))
    # When the agent is INSIDE, the time of day / danger relationship is :
    a1[:,:,0] = np.array([
        [0.25 , 0.75], #morning
        [0.00 , 1.00], #day
        [0.25 , 0.75], #evening
        [1.00 , 0.00], #night
    ])# danger / safe
    # When the agent is outside, the time of day / danger relationship is the same :
    a1[:,:,1] = np.array([
        [0.25 , 0.75], #morning
        [0.00 , 1.00], #day
        [0.25 , 0.75], #evening
        [1.00 , 0.00], #night
    ])# danger / safe

    # observed = f(danger,refuge)
    a2 = np.zeros((No[1],Ns[0],Ns[1]))
    # When the agent is inside, the observed object / danger relationship is :
    a2[:,:,0] = np.array([
        [0.00,0.00], #food
        [1.00,1.00], #nothing
        [0.00,0.00]  #pred
    ])  # <=> you get nothing, but you don't risk anything, whatever the danger level
    # When the agent is outside, the time of day / danger relationship is NOT the same :
    a2[:,:,1] = np.array([
        [0.90,0.40], #food
        [0.10,0.20], #nothing
        [0.00,0.40]  #pred
    ])  # <=> you risk getting eaten if there's danger but you can gather food if there is not :D

    a3 = np.zeros((No[2],Ns[0],Ns[1]))
    a3[:,0,:] = np.array([
        [1,0], #inside
        [0,1]  #outside
    ])
    a3[:,1,:] = np.array([
        [1,0], #inside
        [0,1]  #outside
    ])
    a = [a1,a2,a3]

    d = [np.array([1.00,0.00]),np.array([0.5,0.5])] # I always start inside, with an uncertain env
    c = [np.array([0,0,0,0]),np.array([2,0,-10]),np.array([0,0])]
    
    b = [normalize(np.ones((Ns[0],Ns[0],1))),np.zeros((Ns[1],Ns[1],2))]
    b[1][:,:,0] = np.array([ # Go inside from any state
        [1.0,1.0],
        [0.0,0.0]
    ])
    b[1][:,:,1] = np.array([ # Go outside from any state
        [0.0,0.0],
        [1.0,1.0]
    ])

    u = np.array([[0,0],[0,1]]).astype(int)    
    e = np.ones((u.shape[0],))

    hide_or_seek_layer = mdp_layer("hide_or_seek_model","model",a,b,c,d,e,u,T,Th,in_seed=seed)
    hide_or_seek_layer.learn_options.learn_a = False
    hide_or_seek_layer.learn_options.learn_b = False
    hide_or_seek_layer.learn_options.learn_d = False
    hide_or_seek_layer.hyperparams.cap_action_explo = 2
    hide_or_seek_layer.hyperparams.cap_state_explo = 2
    hide_or_seek_layer.hyperparams.alpha = 32

    return hide_or_seek_layer

def food_predator_provider(T,Th,seed=None):
    Ns = [4,2] #  the time of day / The agent is either inside or outside
    No = [3,2] #  3 levels of found objects / agent inside vs outside
    
    # State 1 : the time of day
    # State 2 : agent is inside / outside 
    
    # Generated observation 1 : found food / nothing / predator !
    # Generated observation 2 : agent inside or outside

    # observed obj = f(tod,agent_inside)
    a1 = np.zeros((No[0],Ns[0],Ns[1]))
    # When the agent is inside, the observed object / danger relationship is :always nothing
    a1[:,:,0] = np.array([
        [0.00,0.00,0.00,0.00], #food
        [1.00,1.00,1.00,1.00], #nothing
        [0.00,0.00,0.00,0.00], #predator
    ])  # <=> you get nothing, but you don't risk anything, whatever the danger
    # When the agent is outside, the time of day / output relationship is NOT the same :
    a1[:,:,1] = np.array([
        [0.80,0.90,0.80,0.00], #food
        [0.10,0.10,0.10,0.50], #nothing
        [0.10,0.00,0.10,0.50], #predator
    ]) # <=> you risk getting eaten if there's danger but you can gather food if there is not :D

    a2 = np.zeros((No[1],Ns[0],Ns[1]))
    for i in range(Ns[0]):
        a2[:,i,:] = np.array([
            [1,0], #inside
            [0,1]  #outside
        ])
    a = [a1,a2]

    d = [np.array([1,0,0,0]),np.array([1,0])]
    c = [np.array([0,0,0]),np.array([0,0])]
    
    b = [normalize(np.ones((Ns[0],Ns[0],1))),np.zeros((Ns[1],Ns[1],2))]
    b[1][:,:,0] = np.array([ # Go inside from any state
        [1.0,1.0],
        [0.0,0.0]
    ])
    b[1][:,:,1] = np.array([ # Go outside from any state
        [0.0,0.0],
        [1.0,1.0]
    ])

    u = np.array([[0,1],[0,0]]).astype(int)    
    e = np.ones((u.shape[0],))

    food_predator_provider_proc = mdp_layer("food_predator_provider","process",a,b,c,d,e,u,T,Th,in_seed=seed)
    return food_predator_provider_proc


if __name__ == '__main__':
    T = 100
    Th = 4
    NTRIALS = 1

    tod_process = time_of_day_perceptron(T,Th,None,False)
    tod_model = time_of_day_perceptron(T,Th,None,True)

    generate_observation_process = food_predator_provider(T,Th,None)
    hide_seek_model = hide_seek_layer(T,Th,None)

    establish_layerLink(tod_process,tod_model,["o","o"])
    establish_layerLink(tod_process,generate_observation_process,["s.0","s.0"])
    establish_layerLink(tod_model,hide_seek_model,["s_d.0","o_d.0"])
    establish_layerLink(generate_observation_process,hide_seek_model,[["o.0","o.1"],["o.1","o.2"]])
    establish_layerLink(hide_seek_model,generate_observation_process,[["u","u"]])

    hide_seek_task_network = network([generate_observation_process,tod_process,tod_model,hide_seek_model],"hide_or_seek_task_network")

    # print(hide_seek_task_network.run_order)
    # hide_seek_task_network.run()
    # hide_seek_task_network.run()
    # hide_seek_task_network.run()
    # hide_seek_task_network.run()

    stmlist = hide_seek_task_network.run_N_trials(NTRIALS,return_STMs=True)
    print(stmlist[0][1].x)
    print(stmlist[0][1].o)

    # print(stmlist[0][2].o)
    # img = Image.fromarray(stmlist[0][2].x_d*255)
    # img = img.resize((1000,300),resample=Image.NEAREST)
    # img.show()
    # print(np.round(stmlist[0][2].x_d,2))
    t = 0
    lower_model_stm = stmlist[t][2]
    upper_model_stm = stmlist[t][3]

    o_d= upper_model_stm.o_d
    x_d = upper_model_stm.x_d
    for i in range(T):
        print(i)
        # print(lower_model_stm.x_d[:,i])
        print("True TOD : " + str(stmlist[0][1].x_d[...,i]))
        print("Perceived TOD : " +str(np.sum(o_d[...,i],axis=(1,2))))
        print("Perceived current safety : " + str(np.sum(np.round(x_d[...,i],2),axis=1)))
    print("-----------------------------------")
    print(np.round(upper_model_stm.u_d,2))
    print("-----------------------------------")
    print(upper_model_stm.o)