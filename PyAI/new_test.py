import statistics as stat
import numpy as np
from pyai.model.metrics import flexible_kl_dir
from pyai.base.function_toolbox import normalize

a_ = np.ones((5,5)) + 3*np.eye(5)
A_ = np.array([[0,0,0,0,1],
               [0,0,0,1,0],
               [0,0,1,0,0],
               [0,1,0,0,0],
               [1,0,0,0,0]])
print(a_,normalize(a_))
print(stat.mean(flexible_kl_dir(normalize(a_),A_,option='centered')))
