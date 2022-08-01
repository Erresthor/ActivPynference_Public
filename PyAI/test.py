import numpy as np
from pyai.base.function_toolbox import custom_entropy,normalize,custom_novelty_calculation,spm_wnorm
import math 


mean_A = -0.3589

perfect_perception = np.eye(5)
# true perception is affected by the mean of their respective distributions :
true_A = np.zeros((5,5))

def eye_skew_mean(skew,size):
    j = size
    i = size
    for k in range(i):
        index = k + skew
        if (index<=-1):
            true_A[0,k] = 1
        elif (index>=j):
            true_A[-1,k]=1
        else :
            true_minor = math.floor(index)
            true_major = math.floor(index)+ 1
            minor = max(true_minor,0)
            major = min(true_major,j-1)
            distance_between_true_value_and_minor = abs(index-true_minor)
            distance_between_true_value_and_major = abs(index-true_major)
            low_value = 1-distance_between_true_value_and_minor # The further we are from the true value, the less the value (linear)
            high_value = 1-distance_between_true_value_and_major
            true_A[minor,k] = true_A[minor,k] + low_value
            true_A[major,k] = true_A[major,k] + high_value
    return true_A
