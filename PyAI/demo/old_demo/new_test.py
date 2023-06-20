

import numpy as np
from pyai.base.function_toolbox import spm_cross

if __name__ == '__main__':
    u = np.array([0.1,0.9])
    s_t1 = np.array([0.66,0.33,0.01])
    s_t2 = np.array([0.1,0.0,0.9])
    # E(q(s1,s2)) [b] 
    # The following transition is expected to have happenned during this time
    # Column = To
    # Line = From
    action_independent_transition =np.outer(s_t2,s_t1)
    K = spm_cross(action_independent_transition,u)
    print(np.round(action_independent_transition,2))
    print(np.round(K[:,:,1],3))