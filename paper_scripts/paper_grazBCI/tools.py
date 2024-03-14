import actynf
import pickle
import sys,os
import numpy as np
import scipy.stats as scistats
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def to_list_of_one_hots(arr_of_indices,to_shape):
    L = []
    for k,dim in enumerate(to_shape):
        arr = np.zeros((dim,))
        try : 
            arr[arr_of_indices[k]] = 1.0
        except:
            raise ValueError("Invalid index")
        L.append(arr)
    return L

# Saving / loading / extracting simulated data
def simulate_and_save(my_net,savepath,Nsubj,Ntrials,override=False):
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))

    exists = os.path.isfile(savepath)
    if (not(exists)) or (override):
        stm_subjs = []
        weight_subjs = []
        print("Saving to " + savepath)
        for sub in range(Nsubj):
            subj_net = my_net.copy_network(sub)

            STMs,weights = subj_net.run_N_trials(Ntrials,return_STMs=True,return_weights=True)
            stm_subjs.append(STMs)
            weight_subjs.append(weights)

        save_this = {
            "stms": stm_subjs,
            "matrices" : weight_subjs
        }
            
        with open(savepath, 'wb') as handle:
            pickle.dump(save_this, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved to :   " + savepath)

def save_output(stm_subjs,weight_subjs,savepath):
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))
    save_this = {
            "stms": stm_subjs,
            "matrices" : weight_subjs
    }
    with open(savepath, 'wb') as handle:
        pickle.dump(save_this, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved to :   " + savepath)

def extract_training_data(savepath):
    # EXTRACT TRAINING CURVES    
    with open(savepath, 'rb') as handle:
        saved_data = pickle.load(handle)
    stms = saved_data["stms"]
    weights = saved_data["matrices"]

    Nsubj = len(stms)
    Ntrials = len(weights[0])-1 # One off because we save the initial weights (= trial 0)
    return stms,weights,Nsubj,Ntrials

def save_object_to(obj,savepath,override=True):
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))

    exists = os.path.isfile(savepath)
    if (not(exists)) or (override):
        print("Saving to " + savepath)
            
        with open(savepath, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved to :   " + savepath)

def color_spectrum(fromcol,tocol,t):
    return fromcol + t*(tocol-fromcol)


# Generating initial weights
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
    return actynf.normalize(discretized_pdf[1:-1])

def gaussian_from_distance_matrix(No,matrix_of_norm_distances,sigma):
    dependent_state_dims = matrix_of_norm_distances.shape
    
    a0 = np.zeros((No,)+dependent_state_dims)        
    projected_to_outcome_space = (1.0 - matrix_of_norm_distances)*(No-1)  # High outcomes for low distances
    for index, mu in np.ndenumerate(projected_to_outcome_space):
        feedback_for_this_state = gaussian_to_categorical(np.zeros(No,),mu,sigma)
        state_obs_dist = (slice(None),)+index       
        a0[state_obs_dist] = feedback_for_this_state
    return a0


# Plotting helpers
def clever_running_mean(arr, N):
    """ 
    For regularly spaced points only
    """    
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

# When the interval between points is not constant, very messy and may need to be rewritten
def clever_running_mean_mess(X,Y,window):
    """ 
    Order the points of a point cloud and smooth them.
    """    
    sorted_vals = np.array(sorted(zip(X,Y)))
    sorted_x = sorted_vals[:,0]
    sorted_y = sorted_vals[:,1]

    ys = uniform_filter1d(sorted_y, size=window)

    squared_error = np.sqrt((sorted_y - ys)*(sorted_y - ys))
    filtered_error = uniform_filter1d(squared_error,size=window)
    return sorted_x,ys,filtered_error

def pointcloud_mean(x,y,win):
    return clever_running_mean_mess(x,y,win)

# KL divergence between two categorical distributions
def dist_kl_dir(a,b,
            return_scalar = True):
    """ The same as scipy.stats.entropy"""
    a = actynf.normalize(a)
    b = actynf.normalize(b)

    if (type(a)==list):
        assert type(b)==list,"a and b should be the same type."
        divs = [dist_kl_dir(ai,bi) for ai,bi in zip(a,b)]
        if (return_scalar):
            return sum(divs)
    else :
        assert a.shape == b.shape,"a and b should be numpy arrays of same shape."
        divs = np.sum(a*np.log(1e-10 + (a/(b+1e-5))))
    return divs

# Jensen-Shannon divergence between two categorical distributions
def js_dir(a,b,return_scalar=True,eps=1e-10):
    a = actynf.normalize(a)
    b = actynf.normalize(b)
    if (type(a)==list):
        assert type(b)==list,"a and b should be the same type."
        divs = [js_dir(ai,bi) for ai,bi in zip(a,b)]
        if (return_scalar):
            return sum(divs)
    else :
        assert a.shape == b.shape,"a and b should be numpy arrays of same shape."
        a[a<eps] = eps
        b[b<eps] = eps # Avoid overflows
        divs = np.sum(jensenshannon(a,b,axis=0))
    return divs


# Plotting tool : 
def imshow_with_labels(ax,array2d,fontsize = 10,vmin=0,vmax=1,normmat = True,overwriteColor = None,roundto=2):
    if normmat:
        im = ax.imshow(actynf.normalize(array2d),vmin=vmin,vmax=vmax)
    else : 
        im = ax.imshow(array2d)
    ax.set_axis_off()

    for i in range(array2d.shape[0]):
        for j in range(array2d.shape[1]):
            if (overwriteColor != None):
                ax.text(j, i, round(array2d[i, j], roundto),
                    ha = "center", va = "center", color = overwriteColor,fontsize=fontsize)
            else :
                ax.text(j, i, round(array2d[i, j], roundto),
                    ha = "center", va = "center", color = "w",fontsize=fontsize)
    return ax

if __name__ == "__main__":
    X  = [0,1,2,9,8,7,6]
    Y = [0,1,2,3,4,5,6]
    print(js_dir(X,Y))
