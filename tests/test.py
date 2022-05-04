
import numpy as np
import matplotlib.pyplot as plt
def softmax(X,axis = None,center=True):
    if not(axis==None) :
        if center : 
            x = np.exp(X - np.max(X)) #,axis=axis))
        else :
            x = np.exp(X)
        return x/np.sum(x,axis=axis,keepdims=True)
    else:
        if center :
            x = np.exp(X - np.max(X))
        else :
            x = np.exp(X)
        Y = x/np.sum(x)
    return Y

A = np.array([0.02,0.98,0,0,0])

B = np.array([[0.02,0.5,0,0,0],
            [0.98,0.5,1,1,1]])
C  =np.array([0.02,0.98,0])
C  =np.array([0.02,0.39,0.59])
N = 100
K = np.logspace(-2,2,N)
# A = np.zeros((N,3))
# A_unc = np.zeros((N,3))
# i = 0
# for k in K :
#     #print(k)
#     A[i,:] = (softmax(C**k,0))
#     A_unc[i,:] = (softmax(C**k,0,False))
#     i = i + 1
# plt.plot(K,A)
# plt.plot(K,A_unc)
# plt.show()
a = np.array([0.02,0.39,0.59])
A = np.zeros((N,3))
B = np.zeros((N,3))
base = np.zeros((N,3))
i = 0
for k in K :
    b = C**k
    A[i,:] = b/np.sum(b)
    B[i,:] = softmax(b,0)
    base[i,:] = C
    i = i + 1


plt.plot(K,A[:,0],color="red")
plt.plot(K,B[:,0],color="red",linestyle='--')
plt.plot(K,base[:,0],color="red",linestyle=':')

plt.plot(K,A[:,1],color="green")
plt.plot(K,B[:,1],color="green",linestyle='--')
plt.plot(K,base[:,1],color="green",linestyle=':')

plt.plot(K,A[:,2],color="blue")
plt.plot(K,B[:,2],color="blue",linestyle='--')
plt.plot(K,base[:,2],color="blue",linestyle=':')
plt.vlines(1,0,1,color='black')
plt.xscale("log")
plt.show()




A = np.array([0.25,0.26,0.69,-0.2,-0.5,0])
B = np.array([0.5,0.5,0,0,0,0])

A[B>0]=B[B>0]
print(A)