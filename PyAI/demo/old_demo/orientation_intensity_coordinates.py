from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from pyai.base.function_toolbox import custom_entropy,normalize,custom_novelty_calculation,spm_wnorm
from pyai.base.plotting_toolbox import multi_matrix_plot
import math 
import matplotlib.cm as cm

cmap  = cm.viridis

# LNT GROUP
experimental_values_left = np.array([[0.5,11],
                                     [1,11.3],
                                     [1.5,11.9],
                                     [2,11.5],
                                     [2.5,11.3],
                                     [3,11.9],
                                     [3.5,11.7],
                                     [4,11.1],
                                     [4.5,11.15],
                                     [5,11.3]])

experimental_values_right = np.array([[0.5,10],
                                     [1,10.2],
                                     [1.5,10.5],
                                     [2,10],
                                     [2.5,10.1],
                                     [3,9.9],
                                     [3.5,10.1],
                                     [4,10],
                                     [4.5,9.9],
                                     [5,9.7]])
                                     
# RNT GROUP

experimental_values_left = np.array([[0.5,10],
                                     [1,9.5],
                                     [1.5,9.4],
                                     [2,10.2],
                                     [2.5,9.8],
                                     [3,9.5],
                                     [3.5,9.8],
                                     [4,10.2],
                                     [4.5,10],
                                     [5,9.8]])

experimental_values_right = np.array([[0.5,10],
                                     [1,11.3],
                                     [1.5,11.9],
                                     [2,11.5],
                                     [2.5,11.3],
                                     [3,11.9],
                                     [3.5,11.7],
                                     [4,11.1],
                                     [4.5,11.15],
                                     [5,11.3]])

def feedback(right,left,group='L'):
    if (np.sum(left+right)<1e-8):
        return 0
    if (group=='L'):
        return (left-right)/(left+right)
    elif(group=='R'):
        return (right-left)/(left+right)

# With bigger spaces
N = 250
n = 100
include_attentional_modulation = True
intensities = np.linspace(0,1,N)
orientations = np.linspace(-1,1,N)
outarr = np.zeros((N,N))
thetas = (orientations+1)*(math.pi/4)

os = np.outer(np.ones((N,)),thetas)
iss = np.outer(intensities,np.ones((N,)))

xs = np.outer(intensities,np.cos(thetas))
ys = np.outer(intensities,np.sin(thetas))
formal_aai = (xs-ys)/(xs+ys+1e-8)
theta_aai = np.cos(2*os)/(1+np.sin(2*os))

if (include_attentional_modulation):
    # We can model simply the effect of attentional level on feedback perception by stating x = i*cos/sin(theta)^i (the closer i is to 0, the less informative the feedback)
    for i in range(N):
        for j in range(N):
            alpha=1
            xd = intensities[i]*np.power(np.cos(thetas[j]), intensities[i]**alpha)
            xg = intensities[i]*np.power(np.sin(thetas[j]), intensities[i]**alpha)
            formal_aai[i,j] = (xd-xg)/(xd+xg+1e-8)

values_right = np.linspace(0,1,N)
values_left = np.linspace(0,1,N)

feedback_array_r = np.zeros((N,N))
feedback_array_l = np.zeros((N,N))
color_array_r = np.ones((n,n,4))
color_array_l = np.ones((n,n,4))
polar_r = np.zeros((n,n,4))
polar_l = np.zeros((n,n,4))



for i in range(N):
    for j in range(N):
        xcoord = min(round(xs[i,j]*n),n-1)
        ycoord = min(round(ys[i,j]*n),n-1)
        # "Right"
        value_aai = (1 + formal_aai[i,j])/2
        feedback_array_r[N-ycoord-1,xcoord] = value_aai 
        color_array_r[n-ycoord-1,xcoord] = cmap(value_aai)
        
        polar_r[n-int(n*i/N)-1,n-int(n*j/N)-1] = cmap(value_aai)
        # "left"
        value_aai = (1-formal_aai[i,j])/2
        feedback_array_l[N-ycoord-1,xcoord] = value_aai
        color_array_l[n-ycoord-1,xcoord] = cmap(value_aai)
        
        polar_l[n-int(n*i/N)-1,n-int(n*j/N)-1] = cmap(value_aai)

import matplotlib as mpl
import matplotlib.gridspec as gridspec

# Right/Left
fig= plt.figure()

gs = gridspec.GridSpec(1, 2,width_ratios=[8,1])
super_axes  = [fig.add_subplot(gs[0]),fig.add_subplot(gs[1])]

the_gridspec = super_axes[0].get_subplotspec().get_gridspec()
super_axes[0].remove()

subfig = fig.add_subfigure(the_gridspec[0])
subfigs = subfig.subfigures(2,1)

axes = subfigs[0].subplots(1,2,sharey=True)
axR = axes[0]
im = axR.imshow(color_array_r, vmin=-1, vmax=1)
labels = ["Low","Medium","High"]
axR.set_xticks([0,int(n/2),n-1])
axR.set_xticklabels(labels,style='oblique',color='r')
axR.set_yticks([0,int(n/2),n-1])
axR.set_yticks([n-1,int(n/2),0])
axR.set_yticklabels(labels,style='oblique',color='b')
#axR.plot(np.linspace(0,N-1,N),np.linspace(0,N-1,N),color='black',linewidth=3)
axR.set_title("RNT",fontname="Times New Roman",size=20,fontweight='bold')

axL = axes[1]
left_cartesian = axL.imshow(color_array_l, vmin=-1, vmax=1)
labels = ["Low","Medium","High"]
axL.set_xticks([0,int(n/2),n-1])
axL.set_xticklabels(labels,style='oblique',color='r')
axL.set_yticks([n-1,int(n/2),0])
axL.set_yticklabels(labels,style='oblique',color='b')
axL.set_title("LNT",fontname="Times New Roman",size=20,fontweight='bold')
subfigs[0].suptitle("Cartesian coordinates (Left/Right)",style='normal')

axR.set_xlabel("Right attention",style='italic',color='r')
axL.set_xlabel("Right attention",style='italic',color='r')
axR.set_ylabel("Left attention",style='italic',color='b')
#axL.set_ylabel("Left attention",style='italic',color='b')

#subfigs[0].colorbar(left_cartesian,ax=axL,cmap = cmap)
# orientation/intensity
axes = subfigs[1].subplots(1,2,sharey=True)
axR = axes[0]
im = axR.imshow(polar_r, vmin=-1, vmax=1)

labels = ["Left","Middle","Right"]
axR.set_xticks([0,int(n/2),n-1])
axR.set_xticklabels(labels,style='oblique',color="g")

labels = ["Low","Medium","High"]
axR.set_yticks([n-1,int(n/2),0])
axR.set_yticklabels(labels,style='oblique',color="purple")

axL = axes[1]
left_polar = axL.imshow(polar_l, vmin=-1, vmax=1)
labels = ["Left","Middle","Right"]
axL.set_xticks([0,int(n/2),n-1])
axL.set_xticklabels(labels,style='oblique',color="g")
labels = ["Low","Medium","High"]
axL.set_yticks([n-1,int(n/2),0])
axL.set_yticklabels(labels,style='oblique',color="purple")

axR.set_xlabel("Orientation",style='italic',color="g")
axL.set_xlabel("Orientation",style='italic',color="g")
axR.set_ylabel("Intensity",style='italic',color="purple")
#axL.set_ylabel("Intensity",style='italic',color="purple")


subfigs[1].suptitle("Polar coordinates (Intensity/Orientation)",style='normal')

ax = super_axes[1]
norm = mpl.colors.Normalize(vmin=-1, vmax=1)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
#ax.set_title("Related feedback",loc='right')

#fig.tight_layout()
fig.suptitle("Alpha assymetry based feedback ")
plt.show()