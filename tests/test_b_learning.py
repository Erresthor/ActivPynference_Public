import numpy as np
from function_toolbox import normalize

T = 3
Nf = 2
Ns = [2,4]
Nu = [1,4]
Np = 5

u = np.zeros((Nf,T))

b = []
for i in range(Nf):
    b.append(np.zeros((Ns[i],Ns[i],Nu[i])))
    b[i][:,:,:] = 0.25
print(b)

x = []
for i in range(Nf):
    x.append(np.zeros((Ns[i],T,Np)))
x[0][:,0,:] = np.array([[0.9,0.1],
                        [0.9,0.1],
                        [0.9,0.1],
                        [0.9,0.1],
                        [0.9,0.1]]).T
x[0][:,1,:] = np.array([[0.9,0.1],
                        [0.9,0.1],
                        [0.9,0.1],
                        [0.9,0.1],
                        [0.9,0.1]]).T
x[0][:,2,:] = np.array([[0.9,0.1],
                        [0.9,0.1],
                        [0.9,0.1],
                        [0.9,0.1],
                        [0.9,0.1]]).T                        

x[1][:,0,:] = np.array([[1,0,0,0],
                        [1,0,0,0],
                        [1,0,0,0],
                        [1,0,0,0],
                        [1,0,0,0]]).T

x[1][:,1,:] = np.array([[0.9,0.1,0  ,0  ],
                        [0  ,0.9,0.1,0  ],
                        [0  ,0.9,0  ,0.1],
                        [0  ,0.1,0.9,0  ],
                        [0  ,0.1,0  ,0.9]]).T
x[1][:,2,:] = np.array([[0.9,0.1,0,0],
                        [0,0,0.9,0.1],
                        [0,0,0.1,0.9],
                        [1,0,0,0],
                        [1,0,0,0]]).T

chosen_actions = np.array([[0,0],
                            [1,2]])

#Action / Policy correspondance matrix
V_ = np.zeros((T-1,Np,Nf))
V_[:,:,0]= np.array([[0,0,0,0,0],      # T = 2
                        [0,0,0,0,0]])     # T = 3  row = time point
    #                colums = possible course of action in this modality (0 -->context states)
V_[:,:,1] = np.array([[0,1,1,2,3],      # T = 2
                        [0,2,3,0,0]])     # T = 3  row = time point in this modality (1 -->behavioural states)
    #                colums = possible course of action
V_ = V_.astype(np.int)

def output_action_probability_density(chosen_actions,b):
    output = []
    for factor in range(len(b)):
        output.append(np.zeros((chosen_actions.shape[0],b[factor].shape[1])))  # Size = T-1 x Np
        for t in range(output[factor].shape[0]):
            output[factor][t,chosen_actions[factor,t]]=1
    return output


action_probability_density = output_action_probability_density(chosen_actions,b)
total_change_matrix = []
for f in range(Nf):
    total_change_matrix.append(np.zeros(b[f].shape))
for t in range(1,T): 
    for factor in range(Nf):
        for policy in range(Np):
            print("---------------- " + str(policy) + " -------------------------------")
            print(np.outer(x[factor][:,t,policy],x[factor][:,t-1,policy].T))        
            print(action_probability_density[factor][t-1,:])                        # The following actions were implemented during this t
            #total_change_matrix[factor][]    # Save this action / transition correspondance to the total_change_matrix
            
            transition_for_policy = np.outer(x[factor][:,t,policy],x[factor][:,t-1,policy].T)  # The following transition is expected to have happenned during this t
                                                                                    # if this policy was followed
                                                                                    # Column = To
                                                                                    # Line = From
                                                                                    # 3rd dim = Upon which action ?
            action_implemented = (action_probability_density[factor][t-1,V_[t-1,policy,factor]])  # Was this action implemented ? 1 (yes) / 0 (no)
            print(np.dot(action_implemented,transition_for_policy))


            total_change_matrix[factor][:,:,V_[t-1,policy,factor]] = total_change_matrix[factor][:,:,V_[t-1,policy,factor]] + action_implemented*transition_for_policy

eta = 0.5
for fact in range (Nf):
    total_change_matrix[fact] = total_change_matrix[fact]/np.sum(total_change_matrix[fact])
    b[fact] = total_change_matrix[fact]*eta + b[fact]
B = (normalize(b,0))


print(B[1][:,:,0])
print(B[1][:,:,1])
print(B[1][:,:,2])
print(B[1][:,:,3])
