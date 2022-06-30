from cmath import pi
import numpy as np
from pyai.base.matrix_functions import *

prior_value_a = np.array([1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0])
stringo  = ""

for i in range (prior_value_a.shape[0]):
    for j in range (prior_value_a.shape[0]):
        the_str = str(i*prior_value_a.shape[0]+j)
        stringo = stringo + the_str + " "
        print(i,j)

print(stringo)