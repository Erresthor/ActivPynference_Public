from cmath import pi
import numpy as np
from pyai.base.matrix_functions import *
from pyai.base.function_toolbox import spm_wnorm
A = np.array([[0.76, 0.24, 0.02 ,0.  , 0.  ],
                [0.22, 0.52, 0.22, 0.02, 0.  ],
                [0.02, 0.22, 0.52, 0.22, 0.02],
                [0.,   0.02, 0.22, 0.52, 0.22],
                [0.,   0.,   0.02, 0.24, 0.76]])
A = A*100 + 1
print(A)

print(spm_wnorm(A)*(A>0))