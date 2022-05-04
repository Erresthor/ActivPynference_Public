# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:23:53 2021

@author: cjsan
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from .miscellaneous_toolbox import isNone

def basic_autoplot(x):
    n = x.shape[0]
    plt.plot(np.linspace(1,n,n),x)
    plt.show()

def matrix_plot(fig,matrix,i=1,name="Figure"):
    dims = matrix.ndim

    if (dims < 3):
        ax1 = fig.add_subplot(111)
        ax1.imshow(matrix,interpolation='nearest',cmap=cm.Greys_r)

    if (dims == 3):
        m = matrix.shape[2]
        for factor in range(m):
            axi = fig.add_subplot(i,m,factor+1)
            axi.imshow(matrix[:,:,factor],interpolation='nearest',cmap="gray",vmin=0,vmax=1)
            axi.title.set_text(name+"_"+str(factor))
    return fig


def multi_3dmatrix_plot(matlist,namelist=None,xlab="x-label",ylab="y-label",colmap='gray'):
    m = matlist[0].shape[2]
    
    assert matlist[0].ndim >1," Matrix dimension > 2"
    
    fig,axes = plt.subplots(nrows = len(matlist),ncols= m)

    cmap = cm

    counter = 1
    legend_holder = int(len(matlist)/2.0)

    for i in range(len(matlist)):
        for factor in range(m):
            axi = axes[i,factor]

            dims = matlist[i].ndim
            im = axi.imshow(matlist[i][:,:,factor],interpolation='nearest',cmap=colmap,vmin=0,vmax=1)
            
            if (not(isNone(namelist))):
                font = {'family': 'serif',
                    'color':  'darkred',
                    'weight': 'normal',
                    'size': 6
                }
                axi.set_title(namelist[i] + " (" + str(factor) + ")",fontdict=font)

    legend_holder = int(len(matlist)/2.0)
    for i in range(len(matlist)):
        if (i==legend_holder):
            axes[i,0].set_ylabel(ylab,fontsize=10)
    
    legend_holder = int(m/2.0)
    for i in range(m):
        if (i==legend_holder):
            axes[-1,i].set_xlabel(xlab,fontsize=10)
        
    # for ax in axes[:,0]:
    #     #ax.set(xlabel=xlab)
    #     #ax.set(ylabel=ylab,fontsize=10)
    #     ax.set_ylabel(ylab,fontsize=10)

    for ax in axes.flat:
        ax.label_outer()
    

    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    #fig.show()

    return fig,axes
    
def multi_2dmatrix_plot(matlist,namelist=None,xlab="x-label",ylab="y-label",colmap='gray'):
    
    assert matlist[0].ndim >1," Matrix dimension > 2"
    
    fig,axes = plt.subplots(nrows = len(matlist))
    cmap = cm

    counter = 1
    for i in range(len(matlist)):
        axi = axes[i]

        dims = matlist[i].ndim
        im = axi.imshow(matlist[i],interpolation='nearest',cmap=colmap,vmin=0,vmax=1)
        
        if (not(isNone(namelist))):
            font = {'family': 'serif',
                    'color':  'darkred',
                    'weight': 'normal',
                    'size': 10,
                    }
            axi.set_title(namelist[i],fontdict=font)
        counter += 1

    legend_holder = int(len(matlist)/2.0)
    for i in range(len(matlist)):
        if (i==legend_holder):
            axes[i].set(ylabel=ylab)
    
    
    axes[-1].set_xlabel(xlab,fontsize=10)

    for ax in axes.flat:
        ax.label_outer()
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    #fig.show()
    
    return fig,axes


def multi_1dmatrix_plot(matlist,namelist=None,xlab="x-label",ylab="y-label",colmap='gray'):
    fig,axes = plt.subplots(nrows = len(matlist),figsize=(20,10))
    
    cmap = cm

    counter = 1
    for i in range(len(matlist)):
        axi = axes[i]

        dims = matlist[i].ndim
        
        matlisti = np.expand_dims(matlist[i],0)
        im = axi.imshow(matlisti,interpolation='nearest',cmap=colmap,vmin=0,vmax=1)
        
        if (not(isNone(namelist))):
            font = {'family': 'serif',
                    'color':  'darkred',
                    'weight': 'bold',
                    'size': 16,
                    }
            axi.set_title(namelist[i] ,fontdict=font)
        counter += 1
        
    legend_holder = int(len(matlist)/2.0)
    for i in range(len(matlist)):
        if (i==legend_holder):
            axes[i].set_ylabel(ylab,fontsize=10)
    # for ax in axes:
    #     #ax.set(ylabel=ylab,fontsize=10)
    #     ax.set_ylabel(ylab,fontsize=10)

    axes[-1].set_xlabel(xlab,fontsize=10)
        
    for ax in axes.flat:
        ax.label_outer()
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    #fig.show()
    
    return fig,axes


def multi_matrix_plot(matlist,namelist=None,xlab="x-label",ylab="y-label",colmap='gray'):
    try :
        fig,axes = multi_3dmatrix_plot(matlist,namelist=namelist,xlab=xlab,ylab=ylab,colmap=colmap)
    except Exception as e_3d:
        try :
            fig,axes = multi_2dmatrix_plot(matlist,namelist=namelist,xlab=xlab,ylab=ylab,colmap=colmap)
        except Exception as e_2d:
            try :
                fig,axes = multi_1dmatrix_plot(matlist,namelist=namelist,xlab=xlab,ylab=ylab,colmap=colmap)
            except Exception as e_1d:
                print("Multiple errors !")
                print("3D matrix print error ------------------------------------")
                print(e_3d)
                print("2D matrix print error ------------------------------------")
                print(e_2d)
                print("2D matrix print error ------------------------------------")
                print(e_1d)

                raise Exception("Did not manage to print the input matrix list")
    return fig,axes
    
if __name__=="__main__":
    # A = np.random.random((9,9,3))
    # B = np.random.random((9,9,3))
    A = np.random.random((9,9))
    B = np.random.random((9,9))
    multi_matrix_plot([A,B], ['Real','Subjective'])
    input()