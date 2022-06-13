import numpy as np
from pyai.base.matrix_functions import *

K = 5
A = np.eye(K)

C = (10*np.random.random((K,K))).astype(int)/10

A = normalize(A)
#C = normalize(np.ones(A.shape))
print(np.round(A,2))
# print(C)
C = normalize(C)
ret1  = centered_kl_dir(C,A)
print(ret1)
print(kl_dir(C,A))
print(centered_kl_dir(C,A))
print(jensen_shannon(C,A))