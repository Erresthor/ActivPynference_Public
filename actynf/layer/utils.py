from ..base.miscellaneous_toolbox import isField
import numpy as np

def get_negative_range(axes,max_size):
    negative = tuple()
    for k in range(max_size):
        if not(k in axes):
            negative += (k,)
    return negative

def get_joint_distribution_along(distribution,axes):
    summed_distribution = np.sum(distribution,get_negative_range(axes,distribution.ndim))
    # Depending on the order of the axes, we swap the distribution
    # axes :
    return reorder_axes(summed_distribution,axes)

def reorder_axes(matrix,axes):
    """ 
    Sort axes by ascending order
    Sort matrix axes following the same order
    """
    ax_list = list(axes)
    n = len(ax_list)
    for i in range(n):
        for j in range(n - i - 1):
            if ax_list[j] > ax_list[j + 1]:
                # SWAP J AND J+1
                ax_list[j], ax_list[j + 1] = ax_list[j + 1], ax_list[j]
                # DO THE SAME FOR THE SUMMED_DISTRIBUTION
                matrix = np.swapaxes(matrix,j,j+1)
    return matrix

def check_prompt_shape(dimension_prompt,object_prompt):
    """ 
    Get the shape of the layer component called by a given dimension
    Always returns a list ?
    """
    assert ("o" in dimension_prompt) or ("u" in dimension_prompt) or ("s" in dimension_prompt),dimension_prompt + " is an invalid connection prompt."

    dim_letter = dimension_prompt
    if ("_" in dimension_prompt):
        dim_letter = dimension_prompt.split("_")[0]
    if (dim_letter=="u"):
        dim_letter = "p"
    
    
    shape_prompt = getattr(object_prompt,"N" + dim_letter)
    if (type(shape_prompt) == int):
        shape_prompt = [shape_prompt]
    return shape_prompt

def minus1_in_arr(arr):
    return (True in (np.absolute(arr + 1.0)<1e-10))

def dist_from_1D_outcome(outcomeIndex, No_val):
    return_arr = np.zeros((No_val,))
    return_arr[outcomeIndex] = 1.0
    return return_arr

def dist_from_definite_outcome(outcomeArray,No):
    return_dist_list = []
    for k in range(outcomeArray.shape[0]):
        return_dist_list.append(dist_from_1D_outcome(outcomeArray[k],No[k]))
        # return_dist_list.append(np.zeros((No[k],)))
        # return_dist_list[k][outcomeArray[k]] = 1
    return_dist = np.zeros(tuple(No))
    return_dist[tuple(outcomeArray)]= 1
    return return_dist,return_dist_list


def dist_from_definite_outcome_accross_t(outcomeArray,No_t_shape,t_axis=None):
    """ We assume that the outcome array is of size [N x T]
    Where N is the number of modalities, and T is the number of timesteps"""
    if (not(isField(t_axis))):
        t_axis = len(No_t_shape)-1

    if (outcomeArray.ndim==1):
        outcomeArray = np.expand_dims(outcomeArray,0)
    return__dist = np.zeros(tuple(No_t_shape))

    No = tuple()
    for dimension in range(len(No_t_shape)):
        if dimension != t_axis :
            No = No + (No_t_shape[dimension],)

    for t in range(No_t_shape[t_axis]) :
        return_array_slice = tuple([(t if (i == t_axis) else slice(None)) for i in range(len(No_t_shape))])
        return__dist[return_array_slice] = dist_from_definite_outcome(outcomeArray[:,t],No)[0]
    
    return return__dist

# def get_margin_distribution_along(distribution,axes):
#     marginalized = spm_complete_margin(distribution)
#     return_this = []
#     if type(axes)==tuple:   
#         for index in axes:
#             return_this.append(marginalized[index])
#         return return_this
#     if type(axes)==int:
#         return marginalized[axes]
#     raise NotImplementedError('The type ' + type(axes) + ' is not implemented for this function')

# def get_definite_value_along(array,axes):
#     return_this = []
#     if type(axes)==tuple:   
#         for index in axes:
#             return_this.append(array[index])
#         return return_this
#     if type(axes)==int:
#         return array[axes]
