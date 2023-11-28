import numpy as np, matplotlib.pyplot as plt


def normal(x, mu, sigm):
    return (1.0/(sigm*np.sqrt(2.0*3.1415)))*np.exp(-0.5*np.power((x-mu)/sigm,2))

lims = [0 , 1]

mu_equiibree = 0.5
mu_measured = 0.402
N = 10

# Let's assume the maximum possible standard deviation for the sample
std_dev = 0.5








X = np.linspace(lims[0],lims[1],1000)
Yeq = normal(X,mu_equiibree,std_dev)
Ymeasured = normal(X,mu_measured,std_dev)
plt.plot(X,Ymeasured)
plt.plot(X,Yeq)
plt.show()


T = np.sqrt(N)* (mu_measured - mu_equiibree)/std_dev

print(T)

print(normal(T,0,1))