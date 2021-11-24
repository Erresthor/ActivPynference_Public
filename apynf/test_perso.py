import numpy as np
import random as rand
import matplotlib.pyplot as plt


def run(p,N,T,meet_per_day,initial_contaminations,infection_duration):
    infection_counter = initial_contaminations
    state = np.zeros((N,2))
    state[:initial_contaminations,0] = 1
    np.random.shuffle(state[:,0])
    for t in range(T):
        # Each and every day
        for pop in range(N):
            if(state[pop,0]==1): # If this pop is contamined
                # Every day, contamined people meet around 5 people :
                for k in range(meet_per_day):
                    i_meet_this_guy = int(N*rand.random())
                    if(i_meet_this_guy != pop)and(state[i_meet_this_guy,0]==0):
                        # I contaminate with a probability p
                        contamination = rand.random()
                        if (contamination <= p) :
                            # I contaminate i_meet_this_guy
                            infection_counter += 1
                            state[i_meet_this_guy,0] = 1

                state[pop,1] += 1
                if(state[pop,1]>infection_duration):
                    state[pop,0] = 2
    return infection_counter



def X_runs(X,p):
    T = 365 # there is a total of 36 days in a year
    N = 1500 # There are 1500 persons here, they are either sane (0), contaminated (1)and therefore contagious with a probability of p or immune(2) 
    meet_per_day = 5

    initial_contaminations = 2
    final_contaminations = 149
    infection_duration = 14
    L = []
    for x in range(X):
        L.append(run(p,N,T,meet_per_day,initial_contaminations,infection_duration))
    return(sum(L)/len(L))

K = np.logspace(-3,-0.5,25)
G = []
for k in range(K.shape[0]):
    G.append(X_runs(20,K[k]))

print(K)
print("---")
print(G)
plt.plot(K,G)
plt.xlabel("Probability of transmission")
plt.ylabel("Infections total")
plt.legend()
plt.show()