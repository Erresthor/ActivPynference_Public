# import seaborn as sns
# sns.set(color_codes=True)
import pandas as pd
from audioop import avg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ..base.function_toolbox import normalize

def avg_dist_entropy(matrix,eps=1e-8,MAXIMUM_INFO = 10000):
    """ Calculates the average of the marginal distributions entropy (dist are along axis 0)"""
    matrix = normalize(matrix)
    
    zeromatrix = np.zeros(matrix.shape)
    zeromatrix[matrix<eps] = np.ones(matrix.shape)[matrix<eps]*MAXIMUM_INFO
    
    zeromatrix[matrix>=eps] = zeromatrix[matrix>=eps] - np.log(matrix[matrix >= eps])

    marginal_entropy = zeromatrix*matrix
    matrix_entropy = np.sum(marginal_entropy,axis = 0)

    return (np.average(matrix_entropy))

def flexible_entropy(matrix_or_list,norm=True):
    if (type(matrix_or_list)==list):
        out = []
        for k in range(len(matrix_or_list)):
            out.append(flexible_entropy(matrix_or_list[k]))
        return out
    elif (type(matrix_or_list)==np.ndarray):
        normalizer = 1
        if (norm):
            normalizer = np.log(matrix_or_list.shape[0])
            # The entropy of a uniform distribution of the same size
        return avg_dist_entropy(matrix_or_list)/normalizer
    else :
        return 0

def avg_kl_dir(matrix_q,matrix_p,eps=1e-8,MAXIMUM_INFO = 10000):
    assert matrix_q.shape == matrix_p.shape,"A divergence can only be calculated between two arrays of same shape"
    matrix_of_kl_dirs = np.zeros(matrix_q([0,...]).shape)

def matrix_kl_dir(matrix_or_list1,matrix_or_list2,except_axis=None) :
    assert type(matrix_or_list1)==type(matrix_or_list2), "Distributions containers should have the same combination of types to calculate a KL dir"
    # If inputs are list(list(array)) --> works
    # If we have something like list(array(list)) --> Does not work (but is that even possible ?)
    if type(matrix_or_list1)==list:
        out = []
        for k in range(len(matrix_or_list1)):
            out.append(matrix_kl_dir(matrix_or_list1[k],matrix_or_list2[k]))
        return out
    elif type(matrix_or_list1)==np.ndarray:
        # Nice ! that's what we want :
        # Calculate KL dir along one axis ?
        return kl_dir(matrix_or_list1,matrix_or_list2,except_axis=except_axis)
    else : 
        return 0

def kl_dir(P,Q,except_axis=None):
    """ Except axis --> we calculate the kl dir function GIVEN the RV corresponding to the axis    
    --> Axis = 0 does not make much sense ?
    --> Example : axis = 1 : we sum p and q (x| y1 = i, forall y2,forall y3,...) [y1 fixed]
                             we get KLdir[p(.|y1)||q(.|y1)] for y1 variable. (vector of dimension 1)
                             --> How the KLdir varies depending for different values of y1
    --> Axis = None : the global divergence between the two distributions for all conditionnal attachments
    """
    eps = 1e-3
    P = P.astype('float64')
    Q = Q.astype('float64')
    assert type(P)==np.ndarray,"Type should be numpy array, but is " + str(type(P))
    assert type(Q)==np.ndarray,"Type should be numpy array, but is " + str(type(Q))
    Q[Q<eps] = eps
    frac = P/Q
    frac[frac<eps] = eps
    axis_to_sum_on = list(range(frac.ndim))
    if(except_axis!=None):
        if(type(except_axis)==int):
            axis_to_sum_on.remove(except_axis)
        elif(type(except_axis)==tuple or type(except_axis)==list):
            for value in except_axis:
                axis_to_sum_on.remove(value)
    if (not(0 in axis_to_sum_on)) :
        axis_to_sum_on.insert(0,0)

    return np.sum(P*np.log(frac),tuple(axis_to_sum_on))

def centered_kl_dir(P,Q,except_axis=None):
    """We center the KL divergence by comparing the kl dir between P and Q to the difference between a uniforn distribution and Q"""
    uncentered_dir = kl_dir(P,Q,except_axis=except_axis)
    uniform_dir = kl_dir(normalize(np.ones(P.shape)),Q,except_axis=except_axis)
    return uncentered_dir/uniform_dir

def jensen_shannon(P,Q,except_axis=None):
    M = (P+Q)/2.0
    return (0.5*kl_dir(P,M,except_axis) + 0.5*kl_dir(Q,M,except_axis))

def flexible_kl_dir(P,Q,except_axis=None,option=None):
    assert type(P)==type(Q), "You should only compare objects of the same type, not "+ str(type(P)) + " and " + str(type(Q))
    if (type(P)==list):
        out = []
        for k in range(len(P)):
            out.append(flexible_kl_dir(P[k],Q[k],except_axis,option))
        return out
    elif (type(P)==np.ndarray):
        if(option=='centered'):
            return centered_kl_dir(P,Q,except_axis)
        elif(option=='jensen-shannon'):
            return jensen_shannon(P,Q,except_axis)
        else :
            return kl_dir(P,Q,except_axis)
    else : 
        return 0
