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

datapath  = "D:\\data\\neuromodels\\indivrun\\indivrun2\\"

K = 0

all_ticks = False

As_list = [] 
Bs_list = []
s_list = []
evals = []
s_dist = []
ulist = []


for filename in os.listdir(datapath):
    exp = load_layer_sumup(os.path.join(datapath, filename))
    if (all_ticks):
        for t in range(len(exp.ticklist)):
            #print(str(exp.k) + " - " + str(exp.ticklist[t].t))
            K += 1
            if (isField(exp.ticklist[t].a_)) :
                As_list.append([exp.ticklist[t].A_[0],exp.ticklist[t].a_[0]])
            else :
                As_list.append([exp.ticklist[t].A_[0],exp.ticklist[t].A_[0]])
            Bs_list.append([exp.ticklist[t].B_[0],exp.ticklist[t].b_[0]])
            #evals.append(evaluate_timestamp(t,exp.ticklist[t].efficacy, exp.ticklist[t].o)[0])
            evals.append(evaluate_timestamp(t,exp.ticklist[t].efficacy, exp.ticklist[t].o)[0])
            s_list.append(exp.ticklist[t].s[0,t])

            
            s_belief = exp.ticklist[t].Q[t,:]
            s_real = np.zeros(s_belief.shape)
            s_real[exp.ticklist[t].s[0,t]] = 1

            #print(matrix_distance(s_real, s_belief))
            s_dist.append(matrix_distance(s_real,s_belief))
            #print(exp.ticklist[t].u)
            if(t<exp.ticklist[t].T-1) :
                ulist.append(exp.ticklist[t].u[0,t])
            else :
                ulist.append(0)
    # do your stuff
    else :
        layer_final_tick = exp.get_final_state()
        #print(layer_final_tick.t)

        if (isField(layer_final_tick.a_)) :
            As_list.append([layer_final_tick.A_[0],layer_final_tick.a_[0]])
        else :
            As_list.append([layer_final_tick.A_[0],layer_final_tick.A_[0]])
        Bs_list.append([layer_final_tick.B_[0],layer_final_tick.b_[0]])
        #print(evaluate_run(layer_final_tick.efficacy, layer_final_tick.o))
        evals.append(evaluate_run(layer_final_tick.efficacy, layer_final_tick.o)[0])
        s_list.append(layer_final_tick.s[0,-1])

        K +=1
        

        s_belief = layer_final_tick.Q[-1,:]
        s_real = np.zeros(s_belief.shape)
        s_real[layer_final_tick.s[0,-1]] = 1

        #print(matrix_distance(s_real, s_belief))
        s_dist.append(matrix_distance(s_real,s_belief))
        #print(exp.ticklist[t].u)
        
        ulist.append(0)

fig, ax1 = plt.subplots()
X = np.linspace(0,K,K)
normalized_points = np.array(evals)/np.max(np.array(evals))

A_dist = []
B_dist = []
for i in range(len(As_list)):
    A_dist.append(matrix_distance(normalize(As_list[i][1]), As_list[i][0]))
    B_dist.append(matrix_distance(normalize(Bs_list[i][1]), Bs_list[i][0]))


ax1.set_ylabel("Behaviour optimality")
ax1.set_xlabel("Iteration")
#ax1.plot(X,normalized_points,'o')
# Real states
ax1.plot(X,s_list/max(s_list),'rx')

# "Optimality"
ax1.plot(X,smooth(normalized_points, 100),'g-',lw=2)

# Complicated and laggy :/
# co = 0
# for filename in os.listdir(datapath):
#     exp = load_layer_sumup(os.path.join(datapath, filename))
#     if co % 2 == 0 :
#         color = 'Blue' # light blue
#     else :
#         color =  'Gray' # light grey
#     ax1.axvspan(co*10,(co+1*10),alpha=0.1,color=color)
#     co = co +1 

ax2 = ax1.twinx()
color = 'tab:black'
ax2.plot(X,A_dist,'b-',label = "A error")
ax2.plot(X,B_dist,'m-',label = "B error")
# ax1.plot(X,s_dist,'r-')
ax2.set_ylabel("Perception error")

#ax2.set_ylim(0,1)

plt.legend()
plt.show()

# interesting_window = [i for i in range(12925,12945)]
# for i in interesting_window:
#     during = matrix_distance(normalize(Bs_list[i-1][1]), Bs_list[i-1][0])
#     after = matrix_distance(normalize(Bs_list[i][1]), Bs_list[i][0])
#     print(str(i) + "  " + str(during) + " --> " + str(after) + "  ( " + str(ulist[i]) + ")")

# print()
# print()
# for u in range(5):
#     r = 3
#     print(27)
#     print(np.round(Bs_list[12927][1][...,u],r))
#     print(28)
#     print(np.round(Bs_list[12928][1][...,u],r))
#     print(29)
#     print(np.round(Bs_list[12929][1][...,u],r))
#     print()
#     print(np.round(Bs_list[12933][1][...,u],r))
#     print(np.round(Bs_list[12934][1][...,u],r))
#     print(np.round(Bs_list[12935][1][...,u],r))
#     print("-----------------")
#     print(27)
#     print(np.round(normalize(Bs_list[12927][1][...,u])),r)
#     print(28)
#     print(np.round(normalize(Bs_list[12928][1][...,u])),r)
#     print(29)
#     print(np.round(normalize(Bs_list[12929][1][...,u])),r)
#     print()
#     print(np.round(normalize(Bs_list[12933][1][...,u])),r)
#     print(np.round(normalize(Bs_list[12934][1][...,u])),r)
#     print(np.round(normalize(Bs_list[12935][1][...,u])),r)
#     print("-----------------")