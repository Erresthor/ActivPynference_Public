from audioop import avg
from unicodedata import unidata_version
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

fig,ax = plt.subplots()

K = 10
N = 10000
Ts = 10*np.linspace(0,N,N+1)/N
Ys = (1 + np.cos(20*Ts))/2.0
# plt.plot(Ts,Ys)
# plt.show()
def animate(i) :
    i = K*i
    i = min(i,N)
    X = Ts[:i]
    Y = Ys[:i]

    ax.clear()
    ax.plot(X,Y)
    
    ax.set_xlim([0,10])
    ax.set_ylim([0,1])
    
    ax.set_ylabel('Motivation')
    ax.set_xlabel('Motivation')
    ax.set_title('Motivation level this day')

ax.legend()
ani = FuncAnimation(fig,animate,frames=int(N/K),interval=10,repeat=True)
plt.show()