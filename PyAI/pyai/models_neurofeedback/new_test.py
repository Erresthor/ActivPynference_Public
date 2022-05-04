import numpy as np
import os,sys,inspect
import random as rand
import matplotlib.pyplot as plt
#from base.function_toolbox import normalize , spm_kron, spm_wnorm, nat_log , spm_psi, softmax , spm_cross
#from pynf_functions import *
import cv2 as cv
import math 
import random 
import noise 
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.ndimage import gaussian_filter


xsize = 400
ysize = 400

phi_arr = np.zeros((xsize,ysize))
m_arr = np.zeros((xsize,ysize))
scale = 50.0

for x in range(xsize):
    for y in range(ysize):
        dist_to_middle = min(math.sqrt((x-xsize/2)*(x-xsize/2) + (y-ysize/2)*(y-ysize/2))/(xsize/2),1)

        k = 1 - dist_to_middle
        #k = 1
        x0 = 1000
        y0 = 1000

        nois = noise.pnoise2((x+x0)/scale, 
                    (y+y0)/scale, 
                    octaves=6, 
                    persistence=0.5, 
                    lacunarity=2,
                    base=0)
        #print(nois)
        r = k*(2*nois + 0.5)
        #print(r)
        #print(dist_to_middle)
        phi_arr[x,y] = r

def dist(A,B):
    return (math.sqrt(np.power(A[0]-B[0],2) + np.power(A[1]-B[1],2)))

interesting_region = (200,150)
distance = 50
vec_arr = np.zeros((xsize,ysize,2))
other_arr = np.zeros((xsize,ysize))
arr = np.zeros((xsize,ysize))
for x in range(xsize):
    for y in range(ysize):
        if (dist(interesting_region,(x,y)) < distance) and (phi_arr[x,y]>0.6):
            arr[x,y] = phi_arr[x,y]
        else :
            other_arr[x,y] = phi_arr[x,y]
        scale = 5
        nois = noise.pnoise2(x/scale, 
                    y/scale, 
                    octaves=2, 
                    persistence=0.5, 
                    lacunarity=2,
                    base=0)

        vector_ystr = math.cos(0.5*math.pi*((2*y-ysize)/ysize))*math.copysign(1, 2*y-ysize) + 4*nois
        vector_xstr = math.cos(0.5*math.pi*((2*x-xsize)/xsize))
        #print(y,vector_str)

        vec_arr[x,y] = np.array([0,vector_xstr*vector_ystr*10])
mask = (arr<=0.75)


fig,ax = plt.subplots(1,1)
im = ax.imshow(gaussian_filter(other_arr,sigma=10),interpolation='nearest')


fig,ax = plt.subplots(1,1)
im = ax.imshow(phi_arr,interpolation='nearest')


fig,ax = plt.subplots(1,1)
im = ax.imshow(arr,interpolation='nearest')



deformed = np.zeros((xsize,ysize))
for x in range(xsize):
    for y in range(ysize):
        nx = (x + vec_arr[x,y,0])
        ny = (y + vec_arr[x,y,1])
        
        x1 = int(nx)
        px1 = nx - x1

        x2 = x1 + 1
        px2 = 1 - px1

        y1 = int(ny)
        py1 = ny - y1
        
        y2 = y1 + 1
        py2 = 1 - py1
        if(x2 < xsize) and (y2 < ysize) and (x1 >= 0) and (y1 >= 0):
            deformed[x1,y1] = deformed[x1,y1] + px1*py1*phi_arr[x,y]
            deformed[x1,y2] = deformed[x1,y2] + px1*py2*phi_arr[x,y]
            deformed[x2,y1] = deformed[x2,y1] + px2*py1*phi_arr[x,y]
            deformed[x2,y2] = deformed[x2,y2] + px2*py2*phi_arr[x,y]
        
scalp_arr = gaussian_filter(deformed,sigma=10)


fig,ax = plt.subplots(1,1)
im = ax.imshow(scalp_arr,interpolation='nearest')

#plt.show()

rescaled_array = rescale(scalp_arr, 0.1,anti_aliasing=False)
fig,ax = plt.subplots(1,1)
im = ax.imshow(rescaled_array,interpolation='nearest')


mu = 0
sigma = 0.1
noise_dist = np.random.normal(mu,sigma,rescaled_array.shape)
fig,ax = plt.subplots(1,1)
im = ax.imshow(noise_dist,interpolation='nearest')


fig,ax = plt.subplots(1,1)
im = ax.imshow(rescaled_array+0.3*noise_dist,interpolation='nearest')













plt.show()
input()