import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import colors as mcolors


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

x=[-1,0,1,2,3,4]
y=[0,1,2,3,4,5]
sigma=0.4
m=1
b=1

Blankz = np.empty([0])
Blanky = np.empty([0])
Blankx = np.empty([0])

for j, i in enumerate(np.arange(-0.5, 2.5, 0.1)):
    Blankz = np.insert(Blankz, j, gaussian(i,1,sigma))
    Blanky = np.insert(Blanky, j, i)
    Blankx = np.insert(Blankx, j, 0)



fig = plt.figure(figsize=(16,6))
ax = fig.gca(projection='3d')
ax.set_xlim(-1,5)
ax.set_ylim(-1,6)
ax.set_zlim(0,2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# create vertice and add it
verts = [list(zip(Blanky, Blankz))]
print(verts)
poly = PolyCollection(verts, facecolors=[mcolors.to_rgba('y', alpha=0.6)])
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=[Blankx[0]], zdir='x')

plt.plot(x,y,0, color="blue",linestyle="dashed")
plt.plot(Blankx,  Blanky, Blankz, color="red")
ax.view_init(60,320)
plt.show()