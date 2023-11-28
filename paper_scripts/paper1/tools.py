import actynf
import numpy as np
import scipy.stats as scistats

def sum_of_values_in_interval(indices,values,bins):
    weighted_sum = np.zeros((bins.shape[0]-1,))
    bin_counter = 0
    for k in range(indices.shape[0]):
        if (indices[k] > bins[bin_counter+1]):
            bin_counter +=1
            # print(bin_counter,bins.shape[0])
            if (bin_counter+1)>=bins.shape[0] :
                return weighted_sum
        try :
            weighted_sum[bin_counter] += values[k]*(indices[k+1]-indices[k])
        except :
            weighted_sum[bin_counter] += values[k]*(indices[k]-indices[k-1])
    return weighted_sum
  
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
    N = array.shape[0]

    min_val = mu-10*sigma
    max_val = mu+10*sigma
    Xs = np.linspace(min_val,max_val,10*N)
    Ys = scistats.norm(mu, sigma).pdf(Xs)

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

    discretized_pdf = sum_of_values_in_interval(Xs,Ys,bins)
    if (option_clamp):
        if option_raw:
            return actynf.normalize(discretized_pdf)   
        clamped_discretization = np.zeros((N,))
        clamped_discretization = discretized_pdf[1:-1]
        clamped_discretization[0] += discretized_pdf[0]
        clamped_discretization[-1] += discretized_pdf[-1]
        return actynf.normalize(clamped_discretization)
    return actynf.normalize(discretized_pdf[1:-1])
