import numpy as np
import os,sys,inspect
import random as rand
import matplotlib.pyplot as plt
from base.function_toolbox import normalize , spm_kron, spm_wnorm, nat_log , spm_psi, softmax , spm_cross
from pynf_functions import *
import cv2 as cv
import math 

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from base.plotting_toolbox import multi_matrix_plot

seed = 0
N = 1000
xsize = 600
ysize = 600
# Let's model each state as an activity spike on a given map position
img = np.zeros((xsize,ysize,3), np.uint8)


random.seed(seed)

def predominance_to_freq(pred):
    # Higher values of predominance mean wider phenomenon
    # Highest being Alpha (1-4Hz)
    # Lowest being Gamma (100Hz)
    return 100*np.exp(-np.log(100)*pred)

def predominance_to_amp(pred,state_value):
    # Amplitude (rated 0 to 1)
    return (0.99*pred+0.01)*state_value

def attenuation(distance, amplitude,attenuation_rate):
    return amplitude*np.power(attenuation_rate,distance)

class state :
    def __init__(self,lvl,init_position):
        self.value = random.random()

        self.predominance = lvl

        self.phi =  2*math.pi*random.random()

        self.position = init_position

        self.value_change_freq = 0.05

        self.update()
        
    def update(self):
        rad = random.random()
        delta = 0.15
        if (rad < self.value_change_freq):
            if (self.value<1):
                self.value = min(self.value + delta,1)
        elif (rad > (1-self.value_change_freq)):
            if (self.value> 0):
                self.value = max(self.value - delta,0)
        
        self.amplitude = predominance_to_amp(self.predominance,self.value)
        
        self.frequency = predominance_to_freq(self.predominance)
        
        self.omega = 2*math.pi*self.frequency



    def draw(self,array):
        xSize = array.shape[0]
        ySize = array.shape[1]

        for i in range(xSize):
            for j in range(ySize):
                dist = math.sqrt(np.power(i-self.position[0],2) + np.power(j-self.position[1],2))
                array[i,j] = array[i,j] + attenuation(dist, self.amplitude, 0.99)
        return array
    
    
X = 300
Y = 300
def random_position(X,Y):
    return (int(random.random()*X),int(random.random()*Y))


state_dist = [[100,0.1],[10,0.4],[2,0.8]]
statelist = []
for k in range(len(state_dist)):
    for l in range(state_dist[k][0]):
        statelist.append(state(state_dist[k][1] + 0.05*(0.5 - random.random()),random_position(X, Y)))

Nit = 10
arraylist = []
for it in range(Nit):
    arr = np.zeros((X,Y))
    for stat in statelist:
        arr = stat.draw(arr)
        stat.update()
    fig,ax = plt.subplots(1,1)

    im = ax.imshow(arr,interpolation='nearest',cmap="plasma")

    #plt.show()

    arraylist.append(np.copy(arr))

# state_1 = state(1,(int(X/2),int(Y/2)))
# state_1.update()
# arr = state_1.draw(arr)
# state_2 = state(0.5,(int(0),int(Y/2)))
# state_2.update()
# arr = state_2.draw(arr)


print(np.sum(arr))
multi_matrix_plot(arraylist, ["" for i in range(len(arraylist))])
input()








