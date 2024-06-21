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

def animal_detector(T,Th,seed=None,isModel=False):
    No = [3,3] # 3 levels of color, 3 levels of textures
    # Grey / Green / white  -   Hairy / Scaly / Feathery
    Ns = 6 # We know 6 different species !
    # 1. Crocodyle (green + scaly), 2. Small  bird (green / white / grey + feathery)
    # 3. Eagle (grey + feathery), 4. Fish (grey/white + scaly)
    # 5. Wolf (grey + hairy), 6. Mutton(grey/white + hairy)
    d = [np.ones((Ns,))] # No prior regarding the species that i encounter

    b = normalize([np.ones((Ns,Ns,1))]) # No prior regarding how animals that i encounter relate to each other
    b[0][:,:,0] = b[0][:,:,0] + np.eye(Ns)

    # b1 = np.array([   # If the species shown depend on the previous one
    #     [0.70,0.02,0.02,0.20,0.02,0.02],#croc
    #     [0.02,0.70,0.20,0.02,0.02,0.02],#bird
    #     [0.02,0.20,0.70,0.02,0.02,0.02],#eagl
    #     [0.20,0.02,0.02,0.70,0.02,0.02],#fish
    #     [0.02,0.02,0.02,0.02,0.70,0.20],#wolf
    #     [0.02,0.02,0.02,0.02,0.20,0.70] #mutton
    # ]) #croc,bird,eagl,fish,wolf,mutton
    # b = normalize([np.expand_dims(b1,-1)])

    a1 = np.array([ # color = f(animal)
        [0.05,0.33,0.90,0.40,0.90,0.50], # Grey
        [0.90,0.33,0.05,0.20,0.05,0.00], # Green
        [0.05,0.33,0.05,0.40,0.05,0.50]  # White
    ])  #croc,bird,eagl,fish,wolf,mutton
    a2 = np.array([ # texture = f(animal)
        [0.05,0.01,0.01,0.10,0.90,0.90], # Hairy 
        [0.90,0.01,0.01,0.80,0.05,0.05], # Scaly 
        [0.05,0.98,0.98,0.10,0.05,0.05]  # Feathery
    ])  #croc,bird,eagl,fish,wolf,mutton
    a = [a1,a2]

    c = [np.array([0,0,0]),np.array([0,0,0])] # We have no intrinsic preferences regarding what color we see

    u = np.array([0])

    e = np.ones(u.shape)

    if isModel:
        animal_detector = mdp_layer("animal_perception_model","model",a,b,c,d,e,u,T,Th,in_seed=seed)
        animal_detector.learn_options.learn_a = False
        animal_detector.learn_options.learn_b = False
        animal_detector.learn_options.learn_d = False
        animal_detector.hyperparams.cap_action_explo = 2
        animal_detector.hyperparams.cap_state_explo = 1
        animal_detector.hyperparams.alpha = 10
    else :
        animal_detector = mdp_layer("animal_perception_process","process",a,b,c,d,e,u,T,Th,in_seed=seed)
    return animal_detector

def anxiety_layer(T,Th,seed=None):
    No = [6] # 6 potential animals
    Ns = [2] # The situation is either dangerous or relaxed

    # animal = f(anxiety), according to our agent, the seen animal is a consequence of how mucg "danger" 
    # there is in the environment
    a1 = np.array([
        [1.0,0.0], # croc
        [0.0,1.0], # bird
        [0.5,0.5], # eagl
        [0.2,0.8], # fish
        [0.8,0.2], # wolf
        [0.0,1.0]  # mutton
    ])  # danger, relaxed
    a = normalize([a1])

    d = [np.array([0.5,0.5])] # I always start inside, with an uncertain env
    c = [np.array([0,0,0,0,0,0])]
    
    b = [normalize(np.ones((Ns[0],Ns[0],1)))]
    b[0][:,:,0] = b[0][:,:,0]  # + np.eye(Ns[0])

    u = np.array([[0,0],[0,1]]).astype(int)    
    e = np.ones((u.shape[0],))

    anxiety_layer = mdp_layer("anxiety_model","model",a,b,c,d,e,u,T,Th,in_seed=seed)
    anxiety_layer.learn_options.learn_a = False
    anxiety_layer.learn_options.learn_b = False
    anxiety_layer.learn_options.learn_d = False
    anxiety_layer.hyperparams.cap_action_explo = 2
    anxiety_layer.hyperparams.cap_state_explo = 2
    anxiety_layer.hyperparams.alpha = 32
    return anxiety_layer

if __name__ == '__main__':
    T = 20
    Th = 4
    NTRIALS = 1

    animal_observator_generator = animal_detector(T,Th,seed=None,isModel=False)
    animal_detector_model = animal_detector(T,Th,seed=None,isModel=True)
    anxiety_model = anxiety_layer(T,Th,seed=None)
     
    establish_layerLink(animal_observator_generator,animal_detector_model,["o","o"])
    establish_layerLink(animal_detector_model,anxiety_model,["x_d","o_d"])
    
    animal_anxiety = network([animal_observator_generator,animal_detector_model,anxiety_model],"animal_anxiety_network")

    stmlist = animal_anxiety.run_N_trials(NTRIALS,return_STMs=True)
    print(stmlist[0][1].x)
    print(stmlist[0][1].o)

    # print(stmlist[0][2].o)
    # img = Image.fromarray(stmlist[0][2].x_d*255)
    # img = img.resize((1000,300),resample=Image.NEAREST)
    # img.show()
    # print(np.round(stmlist[0][2].x_d,2))
    generator_model_stm = stmlist[0][0]
    lower_model_stm = stmlist[0][1]
    upper_model_stm = stmlist[0][2]
    print(np.round(lower_model_stm.x_d,2))

    print(np.round(upper_model_stm.x_d,2))

    print(generator_model_stm.x)
    print(generator_model_stm.o)

    # img = Image.fromarray(lower_model_stm.x_d*255)
    # img = img.resize((1200,800),resample=Image.NEAREST)
    # img.show()

    # img = Image.fromarray(upper_model_stm.x_d*255)
    # img = img.resize((1200,800),resample=Image.NEAREST)
    # img.show()
    fig,axes = plt.subplots(nrows=4,sharex=True,height_ratios=[1, 1, 3,2])
    # axes[0].set_title("Perceived texture")
    axes[0].imshow(np.sum(lower_model_stm.o_d,axis=0),cmap="viridis",aspect="auto")
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels(("HAIRY","SCALY","FEATHERY"))
    
    # axes[1].set_title("Perceived color")
    axes[1].imshow(np.sum(lower_model_stm.o_d,axis=1),cmap="viridis",aspect="auto")
    axes[1].set_yticks(range(3))
    axes[1].set_yticklabels(("GREY","GREEN","WHITE"))
    # axes[0].scatter(np.linspace(0,T-1,T),generator_model_stm.x[0,:],s=10,color="red")
    # axes[0].set_yticks(range(6))
    # axes[0].set_yticklabels(("CROCODYLE","SMALL BIRD","EAGLE","FISH","WOLF","MUTTON"))

    # axes[2].set_title("Perceived animal")
    axes[2].imshow(lower_model_stm.x_d,cmap="viridis",aspect="auto")
    axes[2].scatter(np.linspace(0,T-1,T),generator_model_stm.x[0,:],s=10,color="red")
    axes[2].set_yticks(range(6))
    axes[2].set_yticklabels(("CROCODYLE","SMALL BIRD","EAGLE","FISH","WOLF","SHEEP"))

    # axes[3].set_title("Perceived danger")
    axes[3].imshow(upper_model_stm.x_d,cmap="viridis",aspect="auto")
    axes[3].plot(np.linspace(0,T-1,T),2.0*(upper_model_stm.x_d[1,:]-0.25),linewidth=5,color="red")
    axes[3].set_yticks(range(2))
    axes[3].set_yticklabels(("WORRIED","RELAXED"))
    fig.tight_layout()
    plt.show()