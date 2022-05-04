#!/bin/python
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
import os.path
from scipy import stats
import plotly.graph_objects as go
import matplotlib.pyplot as plt


from ..base.miscellaneous_toolbox import flexible_copy , isField , index_to_dist, dist_to_index
from ..base.function_toolbox import normalize
from ..base.plotting_toolbox import multi_matrix_plot
from ..base.file_toolbox import load_flexible,save_flexible
from ..base.function_toolbox import spm_dot,spm_kron
from ..base.matrix_functions import *
from ..base.file_toolbox import root_path
from ..layer.mdp_layer import mdp_layer
from ..layer.layer_postrun import evaluate_run
from ..layer.layer_learn import MemoryDecayType
from ..layer.layer_sumup import *

from .neurofeedback_base import NF_model_displayer

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
def mse(best_state_succession,state_succession):
    n = best_state_succession.shape[0]
    distance = 0
    for k in range(n):
        distance = distance + m.pow(best_state_succession[k]-state_succession[k],2)
    return (distance / n)

def extract(datapath):
    datapath  = "D:\\data\\neuromodels\\noise_run_2\\"

    noise_level = []
    a_diffs = []
    b_diffs = []
    random_a = []
    random_b = []

    # Final timestep learnt matrices (how much has the subject learnt ?)
    for filename in os.listdir(datapath):
        k = filename
        print(k)
        try :
            k_list = k.split("_")
            nl = float(k_list[0])
            instance = int(k_list[1])
            dirpath = os.path.join(datapath, filename)
            final_file = os.listdir(dirpath)[-1]
            full_final_file = os.path.join(dirpath, final_file)
            exp = load_layer_sumup(full_final_file)


            layer_final_tick = exp.get_final_state()

            a_dif = matrix_distance(normalize(layer_final_tick.A_[0]), normalize(layer_final_tick.a_[0]))
            b_dif = matrix_distance(normalize(layer_final_tick.B_[0]), normalize(layer_final_tick.b_[0]))


            if (nl in noise_level):
                noise_level_index = noise_level.index(nl)
                a_diffs[noise_level_index].append(a_dif)
                b_diffs[noise_level_index].append(b_dif)
            else :
                a_diffs.append([a_dif])
                b_diffs.append([b_dif])
                noise_level.append(nl)
            
        except :
            dirpath = os.path.join(datapath, filename)
            final_file = os.listdir(dirpath)[-1]
            full_final_file = os.path.join(dirpath, final_file)
            exp = load_layer_sumup(full_final_file)

            layer_final_tick = exp.get_final_state()        

            random_a.append(matrix_distance(normalize(layer_final_tick.A_[0]), normalize(layer_final_tick.a_[0])))
            random_b.append(matrix_distance(normalize(layer_final_tick.B_[0]), normalize(layer_final_tick.b_[0])))




    # #POLYNOMIAL FIT
    # # calculate polynomial
    # za = np.polyfit(noise_level, a_arr, 3)
    # zb = np.polyfit(noise_level, b_arr, 3)


    # fa = np.poly1d(za)
    # fb = np.poly1d(zb)

    # #GOMPERTZ FIT
    # from scipy.optimize import curve_fit
    # def gompertz(x,x0,a,k):
    #     return x0*np.exp(k*(1-np.exp(-a*x))/a)

    # def custom(x,x0,k1,A,B,k2,C,D):
    #     return x0 + k1*np.exp(A*(1+B*x) + k2*np.exp(C*(1+D*x)))

    # popt , pcov = curve_fit(gompertz, noise_level, b_arr)
    # print(popt)

    # new_X = np.linspace(noise_level[0],noise_level[-1],1000)
    # a_fitted = fa(new_X)
    # b_fitted = gompertz(new_X,popt[0],popt[1],popt[2])
    # #b_fitted = custom(new_X,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6])
    # #fb(new_X)








    # fig, ax1 = plt.subplots()
    # ax1.set_ylabel("Difference between GT and Believed")
    # ax1.set_xlabel("Noise level")
    # ax1.plot(noise_level,a_arr,'*',label='a-diff')
    # ax1.plot(noise_level,b_arr,'*',label='b-diff')
    # ax1.plot(new_X,a_fitted,'--')
    # ax1.plot(new_X,b_fitted,'--')
    # ax1.plot(noise_level,[np.mean(random_a) for i in range(len(noise_level))],'-',color='purple',label='sham a-diff')
    # ax1.plot(noise_level,[np.mean(random_b) for i in range(len(noise_level))],'-',color='red',label='sham b-diff')
    # plt.legend()
    # plt.show()


    # for filename in os.listdir(datapath)[:10]:
    #     dirpath = os.path.join(datapath, filename)
    #     final_file = os.listdir(dirpath)[-1]
    #     full_final_file = os.path.join(dirpath, final_file)
    #     exp = load_layer_sumup(full_final_file)
    #     layer_final_tick = exp.get_final_state()
    #     multi_matrix_plot([layer_final_tick.B_[0],normalize(layer_final_tick.b_[0])],["Wanted","Real"])
    #     input()

    # Overall performance
    best_state_succession = np.array([0,1,2,3,4,4,4,4,4,4])

    num=25

    noise_level = []
    distlist = []
    randomdistlist = []
    for filename in os.listdir(datapath):
        k = filename
        print(k)
        try :
            k_list = k.split("_")
            nl = float(k_list[0])
            instance = int(k_list[1])
            dirpath = os.path.join(datapath, filename)


            if not(nl in noise_level):
                distlist.append([])
                noise_level.append(nl)
            noise_level_index = noise_level.index(nl)
                
                



            for individual_trial in (os.listdir(dirpath)[num:]) :
                full_filepath = os.path.join(dirpath, individual_trial)
                exp = load_layer_sumup(full_filepath)
                
                layertick = exp.get_final_state()

                dist = mse(best_state_succession,layertick.s[0,:])
                #print(noise_level_index,distlist)
                distlist[noise_level_index].append(dist)

        except :
            dirpath = os.path.join(datapath, filename)
            final_file = os.listdir(dirpath)[-1]
            for individual_trial in (os.listdir(dirpath)[num:]) :
                full_filepath = os.path.join(dirpath, individual_trial)
                exp = load_layer_sumup(full_filepath)
                
                layertick = exp.get_final_state()

                dist = mse(best_state_succession,layertick.s[0,:])

                randomdistlist.append([dist])

    a_arr = np.mean(np.array(a_diffs),axis=1)
    b_arr = np.mean(np.array(b_diffs),axis=1)
    print(distlist)
    state_arr = np.mean(np.array(distlist),axis=1)
    print(state_arr)



    fig, ax1 = plt.subplots()
    ax1.set_ylabel("Difference between GT and Believed")
    ax1.set_xlabel("Noise level")
    ax1.plot(noise_level,state_arr,'*',label='mean performance')
    ax1.plot(noise_level,[np.mean(np.array(randomdistlist)) for i in range(len(noise_level))],label='sham performance')
    plt.legend()
    plt.show()















