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

randommatrix = normalize(np.random.random((5,5)))
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