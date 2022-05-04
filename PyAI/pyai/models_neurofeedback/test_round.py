import numpy as np
import matplotlib.pyplot as plt

A = np.linspace(0,50,50)/50
B = np.round(2*A)
plt.plot(A,B)
plt.show()