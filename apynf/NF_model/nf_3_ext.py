# -*- coding: utf-8 -*-
"""
Created on Tue Aug 3 10:55:21 2021

@author: cjsan
"""
import numpy as np
import random
from PIL import Image, ImageDraw
import sys,os,inspect
import math
import random as r

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from base.spm12_implementation import MDP 
from base.miscellaneous_toolbox import flexible_copy , isField
from base.function_toolbox import normalize
from mdp_layer import mdp_layer
from neurofeedback_base import NF_model_displayer
from base.plotting_toolbox import multi_matrix_plot
from base.file_toolbox import load_flexible,save_flexible
import matplotlib.pyplot as plt

from base.function_toolbox import spm_dot,spm_kron

from layer_postrun import evaluate_run,evaluate_timestamp
from layer_learn import MemoryDecayType
from pynf_functions import *

from layer_sumup import *

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

datapath  = "D:\\data\\neuromodels\\noise_run_2\\"

noise_level = []
adiffs = []
bdiffs = []

for filename in os.listdir(datapath):
    k = filename
    print(k)
    try :
        noise_level.append(float(k))
        
        dirpath = os.path.join(datapath, filename)
        final_file = os.listdir(dirpath)[-1]
        print(final_file)
        #exp = load_layer_sumup()
        full_final_file = os.path.join(dirpath, final_file)
        exp = load_layer_sumup(full_final_file)
        #print(exp)
        layer_final_tick = exp.get_final_state()
        print(layer_final_tick.t)

        adiffs.append(matrix_distance(normalize(layer_final_tick.A_[0]), normalize(layer_final_tick.a_[0])))
        bdiffs.append(matrix_distance(normalize(layer_final_tick.B_[0]), normalize(layer_final_tick.b_[0])))
    except :
        dirpath = os.path.join(datapath, filename)
        final_file = os.listdir(dirpath)[-1]
        full_final_file = os.path.join(dirpath, final_file)
        exp = load_layer_sumup(full_final_file)


        random_a = matrix_distance(normalize(layer_final_tick.A_[0]), normalize(layer_final_tick.a_[0]))
        random_b = matrix_distance(normalize(layer_final_tick.B_[0]), normalize(layer_final_tick.b_[0]))
fig, ax1 = plt.subplots()
ax1.set_ylabel("Difference between GT and Believed")
ax1.set_xlabel("Noise level")
ax1.plot(noise_level,adiffs,'-',label='a-diff')
ax1.plot(noise_level,bdiffs,'-',label='b-diff')
ax1.plot(noise_level,[random_a for i in range(len(noise_level))],'-',color='purple',label='sham a-diff')
ax1.plot(noise_level,[random_b for i in range(len(noise_level))],'-',color='red',label='sham b-diff')
plt.legend()
plt.show()


for filename in os.listdir(datapath)[:10]:
    dirpath = os.path.join(datapath, filename)
    final_file = os.listdir(dirpath)[-1]
    full_final_file = os.path.join(dirpath, final_file)
    exp = load_layer_sumup(full_final_file)
    layer_final_tick = exp.get_final_state()
    multi_matrix_plot([layer_final_tick.B_[0],normalize(layer_final_tick.b_[0])],["Wanted","Real"])
    input()