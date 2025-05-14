from dis import dis
import numpy as np
from scipy.special import gammaln
from scipy.special import psi
import random as random


def sample_index(rng,p,N,replace=False):
    i = rng.choice(np.arange(p.size),size=N, p=p.ravel(),replace=replace)
    # i = rng.sample(np.arange(p.size), p=p.ravel(),replace=replace)
    return np.squeeze(np.stack(np.unravel_index(i, p.shape),axis=-1))

def prune_tree_auto(distribution,random_number_generator, N=None,
                    plausible_threshold=1.0/16.0,
                    deterministic = True,
                    add_noise=True):
    if N == None:
        return np.stack(np.where(distribution>plausible_threshold),axis=-1)#[0]
    
    
    add_this = np.zeros_like(distribution)
    if (add_noise):
        numpy_rng = np.random.default_rng(int(1e10*random_number_generator.random()))
        add_this = 1e-10*np.abs(numpy_rng.normal(size=distribution.shape))
    prune_this = (distribution + add_this).ravel()
    
    
    # Sorting and picking the top N branches:
    max_branches = min(N,prune_this.shape[0])
    if (deterministic):
        sorted_distribution_indices = np.argsort(-prune_this)[:max_branches]
            # Bigger probabilities first
    else :
        numpy_rng = np.random.default_rng(int(1e10*random_number_generator.random()))
        sorted_distribution_indices = sample_index(numpy_rng,prune_this,max_branches,False)

    # Pruning based on plausibility :
    before_plausible_pruning = prune_this[sorted_distribution_indices]
    within_plausibility_indices = sorted_distribution_indices[before_plausible_pruning>=plausible_threshold]
    
    # Transform back to the original distribution shape :
    unraveled_indices = np.unravel_index(within_plausibility_indices,distribution.shape)
    pruned_indices = np.stack(unraveled_indices,axis=-1)

    return pruned_indices  # Always 2Dimensionnal : Nbranches x Ndim    
    
    
    
if __name__ == "__main__":
    test = np.array([[0.0 ,0.8 ,0.0,0.16],
                    [0.00,0.00,0.02,0.02]]).ravel()
    # numpy_rng = np.random.default_rng(200)
    # print(sample_index(numpy_rng,test,3,replace=False))

    rng = random.Random(56)
    print(prune_tree_auto(test,rng,N=2,deterministic=True)[:,0])
    print(prune_tree_auto(test,rng,N=0,deterministic=False)[:,0])
    print("###")
    print(prune_tree_auto(test,rng,N=None,deterministic=False)[:,0])