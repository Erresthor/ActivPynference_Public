# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:23:53 2021

@author: cjsan
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from miscellaneous_toolbox import isNone

def basic_autoplot(x):
    n = x.shape[0]
    plt.plot(np.linspace(1,n,n),x)
    plt.show()

#x = np.random.randint(0,10,(25,))

#basic_autoplot(x)

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


def multi_3dmatrix_plot(matlist,namelist=None,xlab="x-label",ylab="y-label",colmap='gray',vmax=1):
    m = matlist[0].shape[2]
    fig,axes = plt.subplots(nrows = len(matlist),ncols= m)
    counter = 1
    for i in range(len(matlist)):
        for factor in range(m):
            axi = axes[i,factor]

            dims = matlist[i].ndim
            im = axi.imshow(matlist[i][:,:,factor],interpolation='nearest',cmap=colmap,vmin=0,vmax=vmax)
            
            if (not(isNone(namelist))):
                axi.title.set_text(namelist[i]+"_"+str(factor))
            counter += 1

    for ax in axes[:,0]:
        ax.set(ylabel=ylab)
        
    for ax in axes[1,:]:
        ax.set(xlabel=xlab)

    for ax in axes.flat:
        ax.label_outer()
    

    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.show()

def multi_2dmatrix_plot(matlist,namelist=None,xlab="x-label",ylab="y-label",colmap='gray',vmax=1):
    fig,axes = plt.subplots(nrows = len(matlist))

    cmap = cm

    counter = 1
    for i in range(len(matlist)):
        axi = axes[i]

        dims = matlist[i].ndim
        im = axi.imshow(matlist[i],interpolation='nearest',cmap=colmap,vmin=0,vmax=vmax)
        
        if (not(isNone(namelist))):
            axi.title.set_text(namelist[i])
        counter += 1

    for ax in axes:
        ax.set(ylabel=ylab)

    axes[-1].set(xlabel=xlab)

    for ax in axes.flat:
        ax.label_outer()
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.show()

def multi_matrix_plot(matlist,namelist=None,xlab="x-label",ylab="y-label",colmap='gray',vmax=1):
    try :
        multi_3dmatrix_plot(matlist,namelist=namelist,xlab=xlab,ylab=ylab,colmap=colmap,vmax=vmax)
    except Exception as e_3d:
        try :
            multi_2dmatrix_plot(matlist,namelist=namelist,xlab=xlab,ylab=ylab,colmap=colmap,vmax=vmax)
        except Exception as e_2d:
            print("Multiple errors !")
            print("3D matrix print error ------------------------------------")
            print(e_3d)
            print("2D matrix print error ------------------------------------")
            print(e_2d)
            raise Exception("Did not manage to print the input matrix list")

if __name__=="__main__":
    # A = np.random.random((9,9,3))
    # B = np.random.random((9,9,3))
    A = np.random.random((9,9))
    B = np.random.random((9,9))
    multi_matrix_plot([A,B], ['Real','Subjective'])
    input()