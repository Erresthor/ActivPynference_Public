import matplotlib.pyplot as plt
import numpy as np

from pyai.base.function_toolbox import normalize


no_prior = normalize(np.ones((5,5)))
perfect_prior = np.eye(5)
my_prior = normalize(np.ones((5,5)) + 5*np.eye(5))
imp_prio = np.array( [[0.7,0.3,0.1,0.0,0.0],
                [0.2,0.4,0.2,0.1,0.0],
                [0.1,0.2,0.4,0.2,0.1],
                [0.0,0.1,0.2,0.4,0.2],
                [0.0,0.0,0.1,0.3,0.7]])
A_ = np.array([[0.65,0.35,0.15,0.05,0.0 ],
                        [0.2 ,0.3 ,0.2 ,0.1 ,0.05],
                        [0.1 ,0.2 ,0.3 ,0.2 ,0.1 ],
                        [0.05,0.1 ,0.2 ,0.3 ,0.2 ],
                        [0.0 ,0.05,0.15,0.35,0.65]])

randommatrix = np.array([[0.44381707, 0.090666  , 0.22791905, 0.16184417, 0.29982722],
                        [0.18230986, 0.22922889, 0.19130031, 0.17993153, 0.24312072],
                        [0.32752526, 0.17171866, 0.02443329, 0.12934256, 0.04807111],
                        [0.0283902 , 0.33540692, 0.20078937, 0.3792379 , 0.14384766],
                        [0.01795762, 0.17297953, 0.35555798, 0.14964385, 0.26513329]])
print(randommatrix)

matlist = [no_prior,my_prior,imp_prio,perfect_prior,A_,randommatrix]
fig,axes = plt.subplots(1,len(matlist))


for k in range(len(matlist)):  
    ax1=axes[k]
    ax1.imshow(matlist[k],vmax=1,vmin=0)
    ax1.set_xticks([])
    ax1.set_xticks([], minor=True)   
    ax1.set_yticks([])
    ax1.set_yticks([], minor=True)
plt.show()