from actynf.base.function_toolbox import normalize,softmax,nat_log
import numpy as np
from PIL import Image
import sys

if __name__ == '__main__':  
    action_posterior = normalize(np.array([0.01,0.2,0.49,0.2,0.1]))
    print(action_posterior)
    print(np.round(softmax(2.0*nat_log(action_posterior)),2))
    # Ru = softmax(self.hyperparams.alpha * nat_log(self.STM.u_d[:,t]))