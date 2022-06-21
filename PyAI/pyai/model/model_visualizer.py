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
from matplotlib.ticker import AutoMinorLocator
import PIL

from ..base.miscellaneous_toolbox import flexible_copy , isField , index_to_dist, dist_to_index
from ..base.function_toolbox import normalize
from ..base.plotting_toolbox import multi_matrix_plot
from ..base.file_toolbox import load_flexible,save_flexible
from ..base.function_toolbox import spm_dot,spm_kron
from ..base.matrix_functions import *
from ..base.file_toolbox import root_path
from ..layer.mdp_layer import mdp_layer,mdp_layer_options
from ..layer.layer_postrun import evaluate_run
from ..layer.layer_learn import MemoryDecayType
from ..layer.layer_sumup import *

from .active_model import ActiveModel
from .active_model_container import ActiveModelSaveContainer
from .active_model_save_manager import ActiveSaveManager


def show_figures(active_model,save_container,realA_override=None,realB_override=None,realD_override=None):
    t = save_container.t
    
    Nfactor = len(active_model.d)
    Nmod = len(active_model.a)
    
    for f in range(Nfactor):
        prior_d = active_model.d[f]
        prior_b = active_model.b[f]

        current_d = save_container.d_[f]
        current_b = save_container.b_[f]
        
        if not(isField(realB_override)) :
            real_B = active_model.B[f]
        else : 
            real_B = realB_override[f]
        
        if not(isField(realD_override)) :
            real_D = active_model.D[f]
        else : 
            real_D = realD_override[f]
    
    for mod in range(Nmod): 
        prior_a = active_model.a[mod]
        current_a = save_container.a_[mod]
        if not(isField(realA_override)) :
            real_A = active_model.A[mod]
        else : 
            real_A = realA_override[mod]
    
    multi_matrix_plot([normalize(real_B),normalize(prior_b),normalize(current_b)],["Real B","Prior b at t=0 ","Learnt b at t=" +str(t) + " "],"FROM states","TO states")
    multi_matrix_plot([normalize(real_A),normalize(prior_a),normalize(current_a)], ["Real A","Prior a at t=0 ","Learnt a at t=" +str(t) + " "],"State (cause)","Observation (consequence)")
    multi_matrix_plot([normalize(real_D),normalize(prior_d),normalize(current_d)], ["Real D","Prior d at t=0 ","Learnt d at t=" +str(t) + " "], "Initial belief","State")
    plt.show(block=False)
    input()
    plt.close()

def colorfunc(colorlist,t,interp = 'linear'):
        n = len(colorlist)
        if (interp=='linear'):
            for i in range(n):
                current_color_prop = (float(i)/(n - 1))
                next_color_prop = (float(i+1)/(n-1))
                if ((t>=current_color_prop) and (t<=next_color_prop)):
                    ti = (t - current_color_prop)/(next_color_prop-current_color_prop)
                    return colorlist[i+1]*ti + colorlist[i]*(1-ti)

def custom_colormap(colormap,in_array,interpolation='linear') :
    """Not very elegant + only designed for 3D matrices :>(  """
    output_array = np.zeros(in_array.shape+colormap[0].shape)
    for x in range(in_array.shape[0]):
        for y in range(in_array.shape[1]):
            output_array[x,y,:] = colorfunc(colormap,in_array[x,y],interp=interpolation)
    return output_array
        
def draw_a_3D_image(matrix,intermatrix_size=0,colormap=[np.array([55,0,55,255]),np.array([0,255,255,255])]): # input in [0,1]
    matrix_shape = matrix.shape
    x = matrix_shape[0]
    y = matrix_shape[1]
    z = matrix_shape[2]
       
    pre_y = y*z + intermatrix_size*(z - 1)
    pre_x = x
    
    colsize = colormap[0].shape[0]
    output_array = np.zeros((pre_x,pre_y,colsize))  # RGB / RGBA
    
    low = 0
    high = x
    for zi in range(z):
        expanded_dims_mat = np.zeros((x,y,4))
        expanded_dims_mat[:,:,0] = 255*matrix[:,:,zi]
        expanded_dims_mat[:,:,1] = 0
        expanded_dims_mat[:,:,2] = 255*matrix[:,:,zi]
        expanded_dims_mat[:,:,3] = 255
        expanded_dims_mat = custom_colormap(colormap,matrix[:,:,zi],'linear')
        
        output_array[:,low:high,:] = expanded_dims_mat
    
        if(high<pre_y-1)and(intermatrix_size>0) :
            # draw intermatrix
            output_array[:,high:high+intermatrix_size,:] = 255*np.array([1,1,1,0])
        
        low = high + intermatrix_size
        high = low + x
        
    return  PIL.Image.fromarray(output_array.astype(np.uint8))

def generate_model_sumup(modelname,savepath,gifs=False,modality_indice = 0 ,factor_indice = 0,adims=(800,800),bdims=(1500,325,1),colmap = [ np.array([0,0,0,255]) , np.array([95,95,180,255]) , np.array([255,239,10,255]) , np.array([255,100,100,255])]) :
    """Generate the sumup figures for a single model layer instance accross all layer instance.
          /----> inst 1 ---> sumup 1
    MODEL -----> inst 2 ---> sumup 2
          \----> inst 3 ---> sumup 3
    """
    loadpath = os.path.join(savepath,modelname)
    width,height = adims[0],adims[1]
    bwidth,bheight,lim = bdims[0],bdims[1],bdims[2]
    A_list = []
    B_list = []
    D_list = []
    
    model = ActiveModel.load_model(loadpath)
    for file in os.listdir(loadpath):
        
        complete_path = os.path.join(loadpath,file)
        is_file = (os.path.isfile(complete_path))
        is_dir = (os.path.isdir(complete_path))

        if (is_dir) :

            if ("_RESULTS" in file) or ("_MODEL" in file):
                print("Ignoring  file : "+ file)
                #ignore this file 
                continue
            print("Generating sumup for file  : "+ file)
            A_list.append([])
            B_list.append([])
            D_list.append([])
            # This is trial results (layer instance)
            layer_instance = int(file)

            listdir_counter = 0
            HARDLIMIT = 1000
            len_dir = len(os.listdir(complete_path))
            for newfile in os.listdir(complete_path):
                L = newfile.split("_")
                trial_counter = int(L[0])
                timestep_counter = 'f'
                cont = ActiveSaveManager.open_trial_container(loadpath,layer_instance,trial_counter,timestep_counter)
                
                # A is matrix of dimensions outcomes x num_factors_for_state_factor_0 x num_factors_for_state_factor_1 x ... x num_factors_for_state_factor_n
                # draw a 3D image is not made for matrix with dimensions >= 4. As a general rule, we take the initial dimensions of the matrix and pick the 0th 
                # Indices of excess dimensions :
                
                if (listdir_counter < HARDLIMIT)or(listdir_counter==len_dir-1): # Either get the HARDLIMIT firsts or the last one
                
                    try :
                        a_image = cont.a_
                    except :
                        a_image = cont.A_
                    A_list[-1].append(a_image)

                    try :
                        b_image = cont.b_
                    except :
                        b_image = cont.B_
                    B_list[-1].append(b_image)
                    
                    
                    try :
                        d_image = cont.d_
                    except :
                        d_image = cont.D_
                    D_list[-1].append(d_image)

            belief_matrices_plots(modelname,savepath,A_list[-1],B_list[-1],D_list[-1],
                    adims=adims,bdims=bdims,
                    colmap = colmap,
                    plot_modality=modality_indice,plot_factor = factor_indice,instance_string=file,plot_gifs=gifs)

def belief_matrices_plots(modelname,savepath,a_list,b_list,d_list,
                    adims=(800,800),bdims=(1500,325,1),
                    colmap = [ np.array([0,0,0,255]) , np.array([95,95,180,255]) , np.array([255,239,10,255]) , np.array([255,100,100,255])],
                    plot_modality=0,plot_factor = 0,instance_string = "GLOBAL",plot_gifs=False) :
    """Generate the sumup figures for a mean of all model layer instance
          /----> inst 1 ---\ 
    MODEL -----> inst 2 -----> mean results ----> general sumup figure
          \----> inst 3 ---/                 /|\
                                              |
                                              |
                                              |
                                              |
                                         This is me !
    """
    loadpath = os.path.join(savepath,modelname)

    width,height = adims[0],adims[1]
    bwidth,bheight,lim = bdims[0],bdims[1],bdims[2]
    
    model = ActiveModel.load_model(loadpath) # Get the model that caused this global result

    HARDLIMIT = 500
    len_dir = len(a_list)
    listdir_counter = 0

    a_im_list = []
    b_im_list = []
    d_im_list = []
    for trial in range(len(a_list)):

        if (listdir_counter < HARDLIMIT)or(listdir_counter==len_dir-1): # Either get the HARDLIMIT firsts or the last one
            listdir_counter += 1

            try :
                while (a_list[trial][plot_modality].ndim > 3):
                    a_list[trial][plot_modality] = a_list[trial][plot_modality][...,0]
                a_mat = a_list[trial][plot_modality]
            except :
                while (model.A_[plot_modality].ndim > 3):
                    model.A_[plot_modality] = model.A_[plot_modality][...,0]
                a_mat = model.A_[plot_modality]
            

            if (a_mat.ndim < 3):
                a_mat = np.expand_dims(a_mat,-1)
            a_image = draw_a_3D_image(normalize(a_mat), lim,colormap =colmap)
            a_resized = a_image.resize((width,height),PIL.Image.Resampling.NEAREST)

            a_im_list.append(a_resized)
            

            try :
                b_image = draw_a_3D_image(normalize(b_list[trial][plot_factor]),lim,colormap=colmap)
            except :
                b_image = draw_a_3D_image(normalize(model.B_[plot_factor]),lim,colormap=colmap)
            b_resized = b_image.resize((bwidth,bheight),PIL.Image.Resampling.NEAREST)
            b_im_list.append(b_resized)
            
            
            try :
                d_image = draw_a_3D_image(np.expand_dims(np.expand_dims(normalize(d_list[trial][plot_factor]),-1),-1), lim,colormap =colmap)
            except :
                d_image = draw_a_3D_image(np.expand_dims(np.expand_dims(normalize(model.D_[plot_factor]),-1),-1), lim,colormap =colmap)
            d_resized = d_image.resize((width,height),PIL.Image.Resampling.NEAREST)
            d_im_list.append(d_resized)


    # SAVING THE RESULTS FOR THE MEAN OF ALL INSTANCES
    # GIF -->
    result_savepath = os.path.join(savepath,modelname,"_RESULTS_"+instance_string)
    if not os.path.exists(result_savepath):
        try:
            os.makedirs(result_savepath)
        except OSError as exc: # Guard against race condition
            raise
        
    if (plot_gifs):
        # GIFS : 
        fi = min(75,len(b_im_list))  # The first frames shown on a slower pace to get better understanding of learning dynamics
        
        savepath_gif = os.path.join(result_savepath,"b__" + str(modelname) + ".gif")
        b_im_list[0].save(savepath_gif,append_images=b_im_list[1:],save_all=True,duration=30,loop=0)
        savepath_gif = os.path.join(result_savepath,"b_" + "first"+str(fi)+"__" + str(modelname) + ".gif")
        b_im_list[0].save(savepath_gif,append_images=b_im_list[1:fi],save_all=True,duration=150,loop=0)

        savepath_gif = os.path.join(result_savepath,"a__" + str(modelname) + ".gif")
        a_im_list[0].save(savepath_gif, format = 'GIF',append_images=a_im_list[1:],save_all=True,duration=30,loop=0)
        savepath_gif = os.path.join(result_savepath,"a_" +"first"+str(fi)+ "__" + str(modelname) + ".gif")
        a_im_list[0].save(savepath_gif, format = 'GIF',append_images=a_im_list[1:fi],save_all=True,duration=150,loop=0)


    # Save final results for the first instance :
    savepath_img = os.path.join(result_savepath,"a_first__" + str(modelname) + ".png")
    a_im_list[0].save(savepath_img)
    savepath_img = os.path.join(result_savepath,"a_final__" + str(modelname) + ".png")
    a_im_list[-1].save(savepath_img)

    # Grab the ground truth perception matrix and make it an image
    while (model.A[plot_modality].ndim > 3):
        model.A[plot_modality] = model.A[plot_modality][...,0]
    a_mat = model.A[plot_modality]
    if (a_mat.ndim < 3):
        a_mat = np.expand_dims(a_mat,-1)
    a_image = draw_a_3D_image(normalize(a_mat), lim,colormap =colmap)
    a_resized = a_image.resize((width,height),PIL.Image.Resampling.NEAREST)
    savepath_img = os.path.join(result_savepath,"a_true__" + str(modelname) + ".png")
    a_resized.save(savepath_img)
    
    savepath_img = os.path.join(result_savepath,"b_first__" + str(modelname) + ".png")
    b_im_list[0].save(savepath_img)
    savepath_img = os.path.join(result_savepath,"b_final__" + str(modelname) + ".png")
    b_im_list[-1].save(savepath_img)

    # Grab the ground truth action matrix and make it an image
    b_image = draw_a_3D_image(normalize(model.B[plot_factor]),lim,colormap=colmap)
    b_resized = b_image.resize((bwidth,bheight),PIL.Image.Resampling.NEAREST)
    savepath_img = os.path.join(result_savepath,"b_true__" + str(modelname) + ".png")
    b_resized.save(savepath_img)

    #Save scale for the first instance :
    savepath_img = os.path.join(result_savepath,"zz_colorscale__" + str(modelname) + ".png")
    N = 500
    img_array = np.linspace(0,1,N)
    img = np.zeros((100,) + img_array.shape + (4,))
    for k in range(N):
        color_array = colorfunc(colmap,img_array[k])
        img[:,k,:] = color_array
    img = PIL.Image.fromarray(img.astype(np.uint8))
    img.resize((800,100))
    img.save(savepath_img)


    B = model.B[plot_factor]
    try :
        b = model.b[plot_factor]
        b_ = b_list[-1][plot_factor]
    except :
        b = B
        b_ = B

    A = model.A[plot_modality]
    try :
        a = model.a[plot_modality]
        a_ = a_list[-1][plot_modality]
    except :
        a = A
        a_ = A
        
    D = model.D[plot_factor]
    try :
        d = model.d[plot_factor]
        d_ = d_list[-1][plot_factor]
    except :
        d = D
        d_ = D
    DPI = 150

    multi_matrix_plot([normalize(B),normalize(b),normalize(b_)],["Real B","Prior b","Learnt b"],"FROM states","TO states")
    savepath_img = os.path.join(result_savepath,"B_sumup__" + str(modelname) + ".png")
    plt.savefig(savepath_img,bbox_inches='tight',dpi=DPI)
    plt.close()

    multi_matrix_plot([normalize(A),normalize(a),normalize(a_)], ["Real A","Prior a","Learnt a"],"State (cause)","Observation (consequence)")
    savepath_img = os.path.join(result_savepath,"A_sumup__" + str(modelname) + ".png")
    plt.savefig(savepath_img,bbox_inches='tight',dpi=DPI)
    plt.close()

    multi_matrix_plot([normalize(D),normalize(d),normalize(d_)], ["Real D","Prior d","Learnt d"], "Initial belief","State")
    savepath_img = os.path.join(result_savepath,"D_sumup__" + str(modelname) + ".png")
    plt.savefig(savepath_img,bbox_inches='tight',dpi=DPI)
    plt.close()
  
def general_performance_plot (savepath,modelname,save_string,trials,a_err,b_err,a_unc,b_unc,error_states,error_behaviour,smooth_window = 5,show=True,asp_ratio=(10,5),
                                figtitle = "untitled") :
    # Mean of error states and behaviour :
    def sliding_window_mean(list_input,window_size = 3):
        list_output = []
        N = len(list_input)
        for trial in range(N):
            mean_value = 0
            counter = 0
            for k in range(trial - window_size,trial + window_size + 1):
                if(k>=0):
                    try :
                        mean_value += list_input[k]
                        counter += 1
                    except :
                        a = 0
                        #Nothing lol
            list_output.append(mean_value/counter)
        return list_output

    state_error_mean = sliding_window_mean(error_states,smooth_window)
    behaviour_error_mean = sliding_window_mean(error_behaviour,smooth_window)


    color1 = 'tab:red'
    color2 = 'tab:blue'
    fig = plt.figure(figsize=asp_ratio)
    ax1 = fig.add_subplot(211)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xlabel('trial')
    ax1.set_ylabel('mean metric entropy', color=color1)
    ax1.set_ylim([-0.1,1.1])
    

    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylabel('mean "centered" kl divergence', color=color2)  # we already handled the x-label with ax1
    ax2.set_ylim(bottom=-0.1)

    l1 = ax1.plot(trials, a_unc, color=color1,label='A entropy',ls='--')
    l2 = ax1.plot(trials, b_unc, color=color1,label='B entropy',ls='-')
    # instantiate a second axes that shares the same x-axis
    
    l3 = ax2.plot(trials, a_err, color=color2,label='A error',ls='--')
    l4 = ax2.plot(trials, b_err, color=color2,label='B error',ls='-')

    ls = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in ls]
    ax1.legend(ls,labs,loc = 0)

    ax1.grid()
    

    # -----------------------------------------------------------------
    color3 = 'yellow'
    color3l = 'orange'
    color4 = 'cyan'
    color4l = 'purple'
    ax3 = fig.add_subplot(212)
    ax3.grid()

    l1 = ax3.plot(trials,error_states,'*',color=color3,label = 'error w.r.t. optimal states')

    ax4 = ax3.twinx()
    l2 = ax4.plot(trials,error_behaviour,'+',color=color4,label = 'error w.r.t. optimal behaviour')
    l3 = ax3.plot(trials,state_error_mean,"-",color=color3l,label = 'error w.r.t. optimal states (smoothed)')
    l4 = ax4.plot(trials,behaviour_error_mean,"--",color=color4l,label = 'error w.r.t. optimal behaviour (smoothed)')

    ls = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in ls]
    ax3.legend(ls,labs,loc = 'best')

    ax3.set_xlabel('trial')
    ax3.set_ylabel('state error', color=color3l)
    ax3.tick_params(axis='y', labelcolor=color3l)
    ax3.set_ylim([-0.1,1.1])
    ax3.set_ylim(ax1.get_ylim()[::-1])

    ax4.set_ylabel('behaviour error', color=color4l)
    ax4.tick_params(axis='y', labelcolor=color4l)
    ax4.set_ylim([-0.1,1.1])
    ax4.set_ylim(ax1.get_ylim()[::-1])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.suptitle(figtitle, fontsize=16,y=1.08)

    save_folder = os.path.join(savepath,modelname,"_RESULTS_" + save_string)
    

    if not os.path.exists(save_folder):
                try:
                    os.makedirs(save_folder)
                except OSError as exc: # Guard against race condition
                    raise
    
    figname = os.path.join(save_folder,"performances")
    plt.savefig(figname,bbox_inches='tight',dpi=1000)
    if(show):
        plt.show()
    else :
        plt.close()


def trial_plot_old(plotfile,plotmean=False,action_labels="alphabet",title=None):
    labelfont = {
        'weight': 'light',
        'size': 8
        }

    hidden_state_factor = 0
    perc_modality = 0

    cont = ActiveModelSaveContainer.load_active_model_container(plotfile)
    eval_cont = evaluate_container(cont)

    T = cont.T
    timesteps = np.linspace(0,T-1,T)
    
    obs = cont.o
    states = cont.s
    acts = cont.u
    beliefs = cont.X
    print("#############")
    for t in range(T-1):
        print(str(np.round(cont.U_post[:,t],2)) + " -> " + str(acts[hidden_state_factor,t]))
    
    Nactions = cont.U_post.shape[0]
    Ns = beliefs[hidden_state_factor].shape[0]
    No = Ns # In the case of observation-hidden state same size spaces


    my_colormap= [np.array([80,80,80,200]) , np.array([39,136,245,200]) , np.array([132,245,39,200]) , np.array([245,169,39,200]) , np.array([255,35,35,200])]
    
    
    state_belief_image = custom_colormap(my_colormap,beliefs[hidden_state_factor])
    mean_beliefs = argmean(beliefs[hidden_state_factor],axis=0)
            # Only pertinent if states of close indices are spacially 
            # linked    

    # Major ticks every 5, minor ticks every 1
    minor_ticks_x = np.arange(0, T, 1)
    major_ticks_x = np.arange(0, T, 1)-0.5
    ticks_actions = np.arange(0, Nactions, 1)





    # BEGIN ! --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    fig = plt.figure(constrained_layout=True)
    
    subfigures = fig.subfigures(1,2,wspace=0.07, width_ratios=[1.7, 1.])

    

    axes = subfigures[0].subplots(2,1)
    ax1 = axes[0]
    ax2 = axes[1]

    # ax3 = fig.add_subplot(111, zorder=-1)
    # for _, spine in ax3.spines.items():
    #     spine.set_visible(False)
    # ax3.tick_params(labelleft=False, labelbottom=False, left=False, right=False )
    # ax3.get_shared_x_axes().join(ax3,ax1)
    # ax3.set_xticks(minor_ticks,major_ticks)
    # minor_locator1 = AutoMinorLocator(2)
    # ax3.xaxis.set_minor_locator(minor_locator1)
    # ax3.grid(which='minor')

    labels = [str("") for i in minor_ticks_x]
    ax1.set_xticks(minor_ticks_x,major_ticks_x)
    minor_locator1 = AutoMinorLocator(2)
    ax1.xaxis.set_minor_locator(minor_locator1)
    ax1.grid(which='minor')
    ax1.set_xticklabels(labels)

    labels = [("t="+str(i)) for i in minor_ticks_x]
    ax2.set_xticks(major_ticks_x,minor=True)
    ax2.set_xticks(minor_ticks_x)
    ax2.set_yticks(ticks_actions)
    ax2.xaxis.grid(True, which='minor')
    ax2.set_xticklabels(labels)
    
    if (action_labels=="alphabet"):
        letters_list =  list(map(chr,range(ord('a'),ord('z')+1)))
        mylabel = letters_list[:Nactions]
        ax2.set_yticklabels(mylabel)
    
    ax1.set(xlim=(0-0.5, T-0.5))
    ax1.imshow(state_belief_image/255.0,aspect="auto")
    ax1.plot(timesteps,states[hidden_state_factor,:],color='black',lw=3)
    if (plotmean):
        ax1.plot(timesteps,mean_beliefs,color='blue',lw=2,ls=":")
    ax1.plot(timesteps,obs[hidden_state_factor,:],color='w',marker="H",linestyle = 'None',markersize=10)
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.set_ylabel("OBSERVATIONS AND PERCEPTION")
    
    
    ax2.set(xlim=(0-0.5, T-0.5))
    action_posterior_image = custom_colormap(my_colormap,cont.U_post)
    ax2.imshow(action_posterior_image/255.0,aspect="auto")
    ax2.plot(timesteps[:-1],acts[hidden_state_factor,:],color='green',marker="*",linestyle = 'None',markersize=10)
    ax2.set_ylim(ax1.get_ylim()[::-1])
    ax2.set_ylabel("ACTIONS")
    ax2.set_xlabel("Timesteps")
    


    for ax in [ax1,ax2]:
        ax.set_anchor('W')
    # fig.tight_layout()
    # fig.show()


    # fig,axes = plt.subplots(2,1)
    subfigs_nested = subfigures[1].subfigures(2,1)
    axes = subfigs_nested[0].subplots(1,2)
    ax4 = axes[0]
    ax6 = axes[1]
    
    try :
        a_mat = cont.a_[perc_modality]
    except:
        a_mat = cont.A_[perc_modality]
    
    while (a_mat.ndim < 3):
        a_mat = np.expand_dims(a_mat,-1)
    a_image = draw_a_3D_image(normalize(a_mat),colormap =my_colormap)
    ax4.imshow(a_image)
    ax4.set_xlabel("States at time t",font=labelfont)
    ax4.set_ylabel("Cause observations at time t",font=labelfont)
    ax4.set_title('Perception model (after learning)', fontsize=10)
    
    
    #Save scale for the first instance :
    N = 250
    img_array = np.linspace(1,0,N)
    img = np.zeros(img_array.shape +(50,) +  (4,))
    for k in range(N):
        color_array = colorfunc(my_colormap,img_array[k])
        img[k,:,:] = color_array
    img = PIL.Image.fromarray(img.astype(np.uint8))
    #img.resize((800,100))
    ax6.imshow(img) 
    ax6.set_title('Color legend', fontsize=10)
    ax6.set_ylabel("Probability density",font=labelfont)
    ax6.set_xticks([])
    ax6.set_yticks([0,N/2.0,N])
    ax6.set_yticklabels(["1.0","0.5","0.0"])

    axes = subfigs_nested[1].subplots(1,1)
    ax5 = axes
    try :
        b_mat = cont.b_[hidden_state_factor]
    except :
        b_mat = cont.B_[hidden_state_factor]
    lim = 0
    b_image = draw_a_3D_image(normalize(b_mat),lim,colormap =my_colormap)
    
    y_ticks = np.arange(0,Ns,1)
    major_ticks = np.arange(0,Nactions,1)*(Ns+lim)-lim + (Ns+lim)/2.0 -0.5
    minor_ticks=[]
    major_ticks = []
    iamhere=0
    for k in range(Nactions):
        major_ticks.append(iamhere-0.5 + Ns/2.0)
        minor_ticks.append(iamhere-0.5)
        iamhere = iamhere + Ns
        minor_ticks.append(iamhere-0.5)
        iamhere = iamhere + lim
    
    labels = [("t="+str(i)) for i in minor_ticks_x]
    ax5.set_xticks(minor_ticks,minor=True)
    ax5.set_xticks(major_ticks)
    ax5.set_yticks(y_ticks)
    ax5.xaxis.grid(True, which='minor')
    if (action_labels=="alphabet"):
        letters_list =  list(map(chr,range(ord('a'),ord('z')+1)))
        mylabel = letters_list[:Nactions]
        ax5.set_xticklabels(mylabel)
    ax5.set_yticklabels(["" for i in range(Ns)])

    ax5.imshow(b_image)
    ax5.set_title('Action model (after learning)', fontsize=10)
    ax5.set_xlabel("Action X leads from states t",font=labelfont)
    ax5.set_ylabel("To states t+1",font=labelfont)
    
    subfigures[0].suptitle('TRIAL HISTORY', fontsize=13)
    subfigures[1].suptitle('SUBJECT MODEL', fontsize=13)
    if (title==None):
        fig.suptitle('Trial sum-up', fontsize='xx-large')
    else :
        fig.suptitle(title, fontsize='xx-large')
    fig.show()


def trial_plot_figure(T,states_beliefs,action_beliefs,
                observations,real_states,actions,
                a_,b_,
                plotmean=False,action_labels="alphabet",title=None) :
    
    labelfont = {
        'weight': 'light',
        'size': 8
        }
    
    timesteps = np.linspace(0,T-1,T)
    
    Nactions = action_beliefs.shape[0]
    Ns = states_beliefs.shape[0]
    No = Ns # In the case of observation-hidden state same size spaces


    my_colormap= [np.array([80,80,80,200]) , np.array([39,136,245,200]) , np.array([132,245,39,200]) , np.array([245,169,39,200]) , np.array([255,35,35,200])]
    
    
    state_belief_image = custom_colormap(my_colormap,states_beliefs)
    mean_beliefs = argmean(states_beliefs,axis=0)
            # Only pertinent if states of close indices are spacially 
            # linked    

    # Major ticks every 5, minor ticks every 1
    minor_ticks_x = np.arange(0, T, 1)
    major_ticks_x = np.arange(0, T, 1)-0.5
    ticks_actions = np.arange(0, Nactions, 1)



    # BEGIN ! --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    fig = plt.figure(constrained_layout=True)
    
    subfigures = fig.subfigures(1,2,wspace=0.07, width_ratios=[1.7, 1.])

    axes = subfigures[0].subplots(2,1)
    ax1 = axes[0]
    ax2 = axes[1]

    labels = [str("") for i in minor_ticks_x]
    ax1.set_xticks(minor_ticks_x,major_ticks_x)
    minor_locator1 = AutoMinorLocator(2)
    ax1.xaxis.set_minor_locator(minor_locator1)
    ax1.grid(which='minor')
    ax1.set_xticklabels(labels)

    labels = [("t="+str(i)) for i in minor_ticks_x]
    ax2.set_xticks(major_ticks_x,minor=True)
    ax2.set_xticks(minor_ticks_x)
    ax2.set_yticks(ticks_actions)
    ax2.xaxis.grid(True, which='minor')
    ax2.set_xticklabels(labels)
    
    if (action_labels=="alphabet"):
        letters_list =  list(map(chr,range(ord('a'),ord('z')+1)))
        mylabel = letters_list[:Nactions]
        ax2.set_yticklabels(mylabel)
    
    ax1.set(xlim=(0-0.5, T-0.5))
    ax1.imshow(state_belief_image/255.0,aspect="auto")
    ax1.plot(timesteps,real_states,color='black',lw=3)
    if (plotmean):
        ax1.plot(timesteps,mean_beliefs,color='blue',lw=2,ls=":")
    ax1.plot(timesteps,observations,color='w',marker="H",linestyle = 'None',markersize=10)
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.set_ylabel("OBSERVATIONS AND PERCEPTION")
    
    
    ax2.set(xlim=(0-0.5, T-0.5))
    action_posterior_image = custom_colormap(my_colormap,action_beliefs)
    ax2.imshow(action_posterior_image/255.0,aspect="auto")
    ax2.plot(timesteps[:-1],actions,color='white',marker="*",linestyle = 'None',markersize=10)
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax2.set_ylabel("ACTIONS")
    ax2.set_xlabel("Timesteps")
    


    for ax in [ax1,ax2]:
        ax.set_anchor('W')
    # fig.tight_layout()
    # fig.show()


    # fig,axes = plt.subplots(2,1)
    subfigs_nested = subfigures[1].subfigures(2,1)
    axes = subfigs_nested[0].subplots(1,2)
    ax4 = axes[0]
    ax6 = axes[1]
    
    a_mat = a_

    a_image = draw_a_3D_image(normalize(a_mat),colormap =my_colormap)
    ax4.imshow(a_image)
    ax4.set_xlabel("States at time t",font=labelfont)
    ax4.set_ylabel("Cause observations at time t",font=labelfont)
    ax4.set_title('Perception model (after learning)', fontsize=10)
    
    
    #Save scale for the first instance :
    N = 250
    img_array = np.linspace(1,0,N)
    img = np.zeros(img_array.shape +(50,) +  (4,))
    for k in range(N):
        color_array = colorfunc(my_colormap,img_array[k])
        img[k,:,:] = color_array
    img = PIL.Image.fromarray(img.astype(np.uint8))
    #img.resize((800,100))
    ax6.imshow(img) 
    ax6.set_title('Color legend', fontsize=10)
    ax6.set_ylabel("Probability density",font=labelfont)
    ax6.set_xticks([])
    ax6.set_yticks([0,N/2.0,N])
    ax6.set_yticklabels(["1.0","0.5","0.0"])

    axes = subfigs_nested[1].subplots(1,1)
    ax5 = axes
    
    b_mat = b_

    lim = 0
    b_image = draw_a_3D_image(normalize(b_mat),lim,colormap =my_colormap)
    
    y_ticks = np.arange(0,Ns,1)
    major_ticks = np.arange(0,Nactions,1)*(Ns+lim)-lim + (Ns+lim)/2.0 -0.5
    minor_ticks=[]
    major_ticks = []
    iamhere=0
    for k in range(Nactions):
        major_ticks.append(iamhere-0.5 + Ns/2.0)
        minor_ticks.append(iamhere-0.5)
        iamhere = iamhere + Ns
        minor_ticks.append(iamhere-0.5)
        iamhere = iamhere + lim
    
    labels = [("t="+str(i)) for i in minor_ticks_x]
    ax5.set_xticks(minor_ticks,minor=True)
    ax5.set_xticks(major_ticks)
    ax5.set_yticks(y_ticks)
    ax5.xaxis.grid(True, which='minor')
    if (action_labels=="alphabet"):
        letters_list =  list(map(chr,range(ord('a'),ord('z')+1)))
        mylabel = letters_list[:Nactions]
        ax5.set_xticklabels(mylabel)
    ax5.set_yticklabels(["" for i in range(Ns)])

    ax5.imshow(b_image)
    ax5.set_title('Action model (after learning)', fontsize=10)
    ax5.set_xlabel("Action X leads from states t",font=labelfont)
    ax5.set_ylabel("To states t+1",font=labelfont)
    
    subfigures[0].suptitle('TRIAL HISTORY', fontsize=13)
    subfigures[1].suptitle('SUBJECT MODEL', fontsize=13)
    if (title==None):
        fig.suptitle('Trial sum-up', fontsize='xx-large')
    else :
        fig.suptitle(title, fontsize='xx-large')
    return fig
