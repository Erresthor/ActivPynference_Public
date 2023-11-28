
import numpy as np

def extrap_diag_2d(arr,assume_extrema=False,periodic=False):
    """With arr a 2d array"""
    assert arr.ndim == 2,"Array should have 2 dimensions, and not " + str(arr.ndim)
    assert arr.shape[0] == arr.shape[1], "Matrix should be a square matrix !"

    Ns = arr.shape[0]
    sum_of_shifted = 0
    for offset in range(-Ns,Ns+1):
        sum_of_shifted += shifted_matrix(arr, offset,assume_extrema,periodic)
    weighted_sum_of_shifted = sum_of_shifted/(2*Ns-1)
    return weighted_sum_of_shifted

def shifted_matrix(arr, offset,clamp_extrema=False,periodic=False):
    if (periodic):
        # If we believe the states are periodic (going "up" from the upper state leads to the lower state)
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
    return np.roll(arr,offset,(0,1))
    # If offset is positive
    # return_mat[offset:,offset:] = arr[]

if __name__ == '__main__':

    diagonalize_this = np.array([
        [0.0,0.0,0.0,0.0,0.7],
        [0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0],
        [0.3,0.0,0.0,0.0,0.0]
    ])
    print(extrap_diag_2d(diagonalize_this))
    print(extrap_diag_2d(diagonalize_this,True))