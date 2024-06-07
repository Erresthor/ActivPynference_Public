import numpy as np
from functools import reduce
import actynf
from actynf.base.function_toolbox import spm_kron # iterative kronecker product

def vectorize_list_of_1darr(x):
    return np.ravel(reduce(np.multiply.outer, x),order="C")

def vectorize_latent_dim(a,b,c,d,e,u):
    # modify the matrices so that only one latent dimension remains, without changing the dynamics inherent to multiple factors
    a_matrices = [np.reshape(a_mod,(a_mod.shape[0],-1),order="C") for a_mod in a]    
    
    # Some way of "compressing" multiple factors into a single matrix 
    # Slightly different from Matlab script, because our kronecker product orders dimension differently
    assert type(b)==list,"b should be a list in order to vectorize"
    bnorm = actynf.normalize(b)
    sum_of_dirichlet_counts = [np.sum(b_fac,axis=(0,)) for b_fac in b] # dirichlet weight for each (s_prev,action) in each factor
    b_flat = []
    Np = u.shape[0]
    Nf = len(b)
    for action_idx in range(Np) :
        b_flat.append(1)
        for factor_idx in range(Nf):
            b_flat[action_idx] = spm_kron(b_flat[action_idx],bnorm[factor_idx][:,:,u[action_idx,factor_idx]])
    b_flattened = np.stack(b_flat,axis=-1)
    # print(sum_of_dirichlet_counts)
    
    # Converting the Dirichlet counts from N 1-Dimensionnal Dirichlet representations (x1 = i,x2 = j,...xN =k) 
    # to a single (1-Dimensionnal) Dirichlet representation encoding all dimensions (X(i,j,...,k)) 
    # is not so forward ...
    # Here, we approximate the resulting dirichlet count as the Nn-th root of the product of the Dirichlet counts
    # this is **not** an exact correspondance between the two distribution but seeks to preserve
    # Zones of low confidence in the distribution (low dirichlet parameter values) while avoiding exploding 
    # dirichlet counts for high confidence areas.
    
    
    b_kron = []
    Np = u.shape[0]
    Nf = len(b)
    for action_idx in range(Np) :
        b_kron.append(1)
        for factor_idx in range(Nf):
            b_kron[action_idx] = spm_kron(b_kron[action_idx],b[factor_idx][:,:,u[action_idx,factor_idx]])
        b_kron[action_idx] = np.power(b_kron[action_idx],1.0/Nf)
    b_matrix = np.stack(b_kron,axis=-1)
    # print(b_matrix[:,0,:])
    for act in range(Np):
        print(np.round(actynf.normalize(b_matrix)[:,:,act],2))
        print(np.round(b_flattened[:,:,act],2))
        print("-------")
    exit()
    
    
    d_matrix = np.ravel(reduce(np.multiply.outer, d),order="C")
    
    # print(d_matrix)
    print([B.shape for B in b])
    print(b_matrix.shape)
    a0 = a[0]
    
    print(b_matrix[:,:,0])
    print(b_matrix[:,:,1])
    print(b_matrix[:,:,2])
    print(b_matrix[:,:,3])
    return a,b,c,d,e,u
    # c does not change