import numpy as np
import math
import enum
import matplotlib.pyplot as plt

def quick_norm(x):
    return x/np.sum(x+1e-10)

def extrap_diag_2d(arr,assume_extrema=False,periodic=False,
                   fade_out_function=(lambda x: 1.0),normalize_fadeout=True):    
    """With arr a 2d array"""
    assert arr.ndim == 2,"Array should have 2 dimensions, and not " + str(arr.ndim)
    assert arr.shape[0] == arr.shape[1], "Matrix should be a square matrix !"
    Ns = arr.shape[0]
    
    offset_array = np.linspace(-Ns,Ns,2*Ns+1).astype(int)
    normalized_absolute_x = np.abs(offset_array)/Ns
    # All possible other states relative to the middle of the matrix

    vectorized_fadeout = np.vectorize(fade_out_function)
    weight_array = vectorized_fadeout(normalized_absolute_x)
    if normalize_fadeout:
        weight_array = quick_norm(weight_array)
    
    sum_of_shifted = 0
    for offset,weight in zip(offset_array,weight_array):
        sum_of_shifted += weight*shifted_matrix(arr, offset,assume_extrema,periodic)
    return sum_of_shifted

def extrapolate_flat(flattened_matrix, latent_shape):
    """ 
    Assuming that this is a transition matrix :
    Flattened matrix is a 2D matrix of size Total_number_of_states x Total_number_of_states
    Total_number_of_states Ntot = product(latent_shape) different states, based on latent subdimensions. 
    Goal : separate this matrix into len(latent_shape) different submatrices
    Then, extrapolate them diagonally using the function above
    Then recompress them into a single matrix
    --> This allows the subject to extrapolate the effect of their actions and map complex observation-state combinations
    """
    # Step 1 : transfom flattened matrix (Ntot x Ntot) into len(latent_shape) matrices of size (latent_shape[i] x latent_shape[i])


    # Step 2 : extrap_2d those matrices


    # Step 3 : integrate into a single matrix (Ntot x Ntot)

def shifted_matrix(arr, offset,clamp_extrema=False,periodic=False):
    if (periodic):
        # If we believe the states are periodic 
        # (going "up" from the uppermost state leads to the lowermost state)
        return np.roll(arr,offset,(0,1))
    
    shifted = np.zeros(arr.shape)
    if offset > 0:
        shifted[offset:, offset:] = arr[:-offset, :-offset]
        # for TO values not comprised in matrix, but with plausible FROM values, we
        # can hypothesize that the transition would result in the maximum / minimum possible state value
        # (NO PERIODICITY HERE, the maximum state does not loop back)
        if (clamp_extrema) :
            plausible_froms = arr[-offset:, :-offset]
            top_vals = np.sum(plausible_froms,axis=0)
            shifted[-1,offset:] += top_vals

    elif offset < 0:
        shifted[:offset, :offset] = arr[-offset:, -offset:]

        if (clamp_extrema) :
            
            plausible_froms = arr[:-offset, -offset:]
            bottom_vals = np.sum(plausible_froms,axis=0)
            shifted[0,:offset] += bottom_vals
    else : 
        return np.copy(arr)
    return shifted

if __name__ == '__main__':

    diagonalize_this = np.array([
        [0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.7,0.0],
        [0.0,0.3,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0]
    ])
    # 
    Ns = 10
    diagonalize_this = np.zeros((Ns,Ns)) 
    diagonalize_this[2,6] = 0.6
    diagonalize_this[1,6] = 0.7
    diagonalize_this[0,2] = 0.4
    fade_out_function=(lambda x: 1.0)
    gen_f = 1.5
    fade_out_function=(lambda x: np.clip(np.exp(-gen_f*x),0.0,1.0))
    plt.imshow(extrap_diag_2d(diagonalize_this,False,fade_out_function=fade_out_function))
    plt.show()
    # def fade_out(distance):
    #     x = abs(distance)


    # fade_out_function=(lambda dist: 1.0)
    fadeout_v = np.vectorize(fade_out_function)
    # froms = np.linspace(-Ns,Ns,2*Ns+1)
    # plt.plot(froms,fadeout_v(froms))
    # plt.show()
    # print(froms)
    # print(double_vec(froms))