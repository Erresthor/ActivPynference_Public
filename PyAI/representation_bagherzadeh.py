from turtle import color, right
import numpy as np
import matplotlib.pyplot as plt
from pyai.base.function_toolbox import custom_entropy,normalize,custom_novelty_calculation,spm_wnorm
from pyai.base.plotting_toolbox import multi_matrix_plot
import math 

def func_aai(r,l):
    if (r+l < 1e-8):
        return 0
    else : 
        print(r,l,(r-l)/(r+l) )
        return (r-l)/(r+l)  

N = 100


def do_figure():
    
    rights = np.linspace(0,1,N)
    lefts = np.linspace(0,1,N)

    rights_lefts_matrix = np.zeros((N,N,2))
    xs = np.zeros((N,N,2))
    Xs = np.zeros((N,N,2))
    for i in range(N):
        for j in range(N):
            xs[i,j,0] = rights[i]
            xs[i,j,1] = lefts[j]
            Xs[i,j,0] = np.sqrt(np.power(rights[i],2)+np.power(lefts[j],2))
            Xs[i,j,1] = aai(rights[i],lefts[j])

    def colormap(t):
        #print(t)
        low = np.array([0,0,1])
        high = np.array([1,0,0])
        return low*(1-t) + (high)*(t)

    def colormap2(t):
        #print(t)
        low = np.array([1,0,1])
        high = np.array([0,1,0])
        return low*(1-t) + (high)*(t)

    fig = plt.figure()
    axes = fig.subplots(2,2)
    ax1 = axes[0,0]
    ax2 = axes[0,1]

    ax3 = axes[1,0]
    ax4 = axes[1,1]


    ax1.set_title("Two states coordinates")
    ax2.set_title("Orientation/Intensity coordinates")
    for i in range(N):
        for j in range(N):
            intensity = Xs[i,j,0]
            if (intensity <= 1):
                #this is a point
                #print(i)
                col = colormap(float(i)/(N-1))
                ax1.plot(xs[i,j,0],xs[i,j,1],color=col,marker=".")
                ax2.plot(Xs[i,j,1],Xs[i,j,0],color=col,marker=".")

                col = colormap2(float(j)/(N-1))
                ax3.plot(xs[i,j,0],xs[i,j,1],color=col,marker=".")
                ax4.plot(Xs[i,j,1],Xs[i,j,0],color=col,marker=".")
    plt.show()

fig = plt.figure()
axes = fig.subplots(1,2)

def angle_aai(theta):
    return np.cos(2*theta)/(1+np.sin(2*theta))

intensities = np.linspace(0,1,N)
orientations = np.linspace(-1,1,2*N)
outarr = np.zeros((N,2*N))
thetas = (orientations+1)*(math.pi/4)

os = np.outer(np.ones((N,)),thetas)
iss = np.outer(intensities,np.ones((2*N,)))

xs = np.outer(intensities,np.cos(thetas))
ys = np.outer(intensities,np.sin(thetas))
formal_aai = (xs-ys)/(xs+ys+1e-8)
theta_aai = np.cos(2*os)/(1+np.sin(2*os))

axes[0].imshow(formal_aai+os)
axes[1].imshow(os)
plt.show()
#         print(theta,Xs[i,j,1])
#         #outarr[i,j] = np.cos(2*theta)/(1+np.sin(2*theta))
        
#         outarr[i,j] = np.cos(2*theta)/(1+np.sin(2*theta))
# axes.imshow(outarr)
# plt.show()