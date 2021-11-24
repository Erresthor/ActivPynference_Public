import numpy as np


from base.function_toolbox import spm_dot,normalize

O = np.array([1,0,0,0,0])
Ns = [5,2]
A_obs_mental = np.zeros((Ns[0],Ns[0],Ns[1]))
    # When attentive, the feedback is modelled as perfect :
A_obs_mental[:,:,0] = np.array([[1,0,0,0,0],
                                [0,1,0,0,0],
                                [0,0,1,0,0],
                                [0,0,0,1,0],
                                [0,0,0,0,1]])
# When distracted, the feedback is modelled as noisy :
A_obs_mental[:,:,1] = np.array([[0.5 ,0.25,0   ,0   ,0   ],
                                [0.5 ,0.5 ,0.25,0   ,0   ],
                                [0   ,0.25,0.5 ,0.25,0   ],
                                [0   ,0   ,0.25,0.5 ,0.5 ],
                                [0   ,0   ,0   ,0.25,0.5]])

print(np.expand_dims(O,-1))
print(spm_dot(A_obs_mental,np.expand_dims(O,-1)))