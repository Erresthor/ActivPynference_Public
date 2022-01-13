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


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from base.spm12_implementation import MDP 
from base.miscellaneous_toolbox import flexible_copy , isField
from base.function_toolbox import normalize
from mdp_layer import mdp_layer
from base.plotting_toolbox import multi_matrix_plot
from base.file_toolbox import load_flexible,save_flexible
import matplotlib.pyplot as plt

from base.function_toolbox import spm_dot,spm_kron

from layer_postrun import evaluate_run

path = "D:\\data\\test\\evals_1.txt"
K = load_flexible(path)
K = np.array(K)
maxi = np.max(K)

normalized_points = K/maxi
N = normalized_points.shape[0]
X = np.linspace(0, N, N)


path = "D:\\data\\test\\diffs.txt"
mats = load_flexible(path)
print(mats)
As = []
Bs = []
for i in range(len(mats)) :
    As.append(mats[i][0])
    Bs.append(mats[i][1])

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

fig, ax1 = plt.subplots()

ax1.set_ylabel("Behaviour optimality")
ax1.set_xlabel("Iteration")
ax1.plot(X,normalized_points,'o')
ax1.plot(X,smooth(normalized_points, 100),'g-',lw=2)


ax2 = ax1.twinx()
color = 'tab:black'
ax2.plot(X,As,'b-',label = "A error")
ax2.plot(X,Bs,'m-',label = "B error")
ax2.set_ylabel("Perception error")

plt.legend()
plt.show()


path = "D:\\data\\test\\b_1.txt"
bs = load_flexible(path)
path = "D:\\data\\test\\a_1.txt"
a_s = load_flexible(path)

k = len(Bs)-1
coun = 0
indice_0 = 0
indice_1 = 0
while ((k>0) and ((indice_0==0) or (indice_1==0))):
    if (Bs[k]>0.7):
        if coun > 3 :
            indice_1 = k
        coun += 1
    if (Bs[k]<0.01):
        indice_0 = k
    k = k - 1

b0 = normalize(bs[indice_0][0])
b1 = normalize(bs[indice_1][0])
multi_matrix_plot([b0,b1], ["Close B","Far B"])

a0 = normalize(a_s[indice_0][0])
a1 = normalize(a_s[indice_1][0])
multi_matrix_plot([a0,a1], ["Close B","Far B"])

print(normalized_points[indice_0],normalized_points[indice_1])

input()
# B = lay.B_[0]
# b = normalize(lay.b_[0],)
# multi_matrix_plot([B,b], ["Real B","Learnt B"])

# A = lay.A_[0]
# a = normalize(lay.a_[0])
# multi_matrix_plot([A,a], ["Real A","Learnt A"])
