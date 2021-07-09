
import numpy as np

A = np.array([-1,-1,2,3,-1])
A[A<0] = 1
print(A)