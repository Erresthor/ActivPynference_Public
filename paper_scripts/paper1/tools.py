import actynf
import numpy as np
import scipy.stats as scistats
import matplotlib.pyplot as plt

def color_spectrum(fromcol,tocol,t):
    return fromcol + t*(tocol-fromcol)

def discretize_dist_between_bins(bins,distr,Nsamples=100):
    discretized = np.zeros((bins.shape[0]-1,))
    for k in range(bins.shape[0]-1):
        # Approximate pdf integral : 
        xs = np.linspace(bins[k], bins[k+1], Nsamples+1)
        ys = distr.pdf(xs)
        
        # There are Nsamples slices of discretized data
        # Each sample occupies (bins[k+1]-bins[k])/Nsamples on the x axis
        h = (bins[k+1]-bins[k])/Nsamples
        individual_slices = 0.5*(ys[0]+ys[-1]) + np.sum(ys[1:-1])
        approximate_density = (individual_slices*h)
        discretized[k] = approximate_density
    return discretized
  
def gaussian_to_categorical(array,               
        mu,sigma,
        array_bins=None, 
        option_clamp = False,option_raw=False):
    """ 
    In :
    - array : empty array of size (N,)
    - array_bins : monotonous array of size (N+1,) a mapping of what indices each cell of array comprises :
        array[k] comprises indices from array_bins[k] to array_bins[k+1]
    - mu : mean of the normal distribution
    - sigma : standard deviation of the normal distribution
    Out : 
    - pdf of the normal distribution projected on the array, assuming the following :
        - The discrete categorical array has a coherent structure (distance between two contiguous indices is the same everywhere +
                    the index of each cell corresponds to the corresponding axis in the normal pdf)
    This means that the gaussian distribution must be relatively centered on [0,N]
    This also means that one needs to apply transformations on the output distribution to make it useful in some cases :)
    """
    distribution = scistats.norm(mu, sigma)

    N = array.shape[0] # Size of the output categorical distribution

    min_val = mu-10*sigma
    max_val = mu+10*sigma
    Xs = np.linspace(min_val,max_val,10*N)
    # .pdf(Xs)
    # plt.plot(Xs,Ys)
    # plt.show()

    bins = np.zeros((N+3,))
    bins[0] = min_val
    bins[-1] = max_val
    
    if (actynf.isField(array_bins)):
        # The user specified a discrete-continuous index mapping !
        bins[1:-1] = array_bins
        if (array_bins[0]<min_val):
            bins[0] = array_bins[0] - 1
        if (array_bins[-1]>max_val):
            bins[-1] = array_bins[-1] + 1
    else :
        # Assume that the index of the distribution corresponds to their
        # continuous value
        bins[1:-1] = np.linspace(-0.5,N-0.5,N+1)
    # print(bins)
    
    discretized_pdf = discretize_dist_between_bins(bins,distribution,100)
    if (option_clamp):
        if option_raw:
            return actynf.normalize(discretized_pdf)   
        clamped_discretization = np.zeros((N,))
        clamped_discretization = discretized_pdf[1:-1]
        clamped_discretization[0] += discretized_pdf[0]
        clamped_discretization[-1] += discretized_pdf[-1]
        return actynf.normalize(clamped_discretization)
    # print(discretized_pdf)
    return actynf.normalize(discretized_pdf[1:-1])

def clever_running_mean(arr, N):
    xarr = np.array(arr)
    xpost = np.zeros(xarr.shape)
    # raw_conv = np.convolve(x, np.ones(N)/N, mode='same')
    for k in range(xarr.shape[0]):
        localmean = 0.0
        cnt = 0.0
        for i in range(k-N,k+N+1,1):
            if ((i>=0) and (i<xarr.shape[0])):
                localmean += xarr[i]
                cnt += 1
        xpost[k] = localmean/(cnt+1e-18)
    return xpost