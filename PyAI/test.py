import numpy as np
from pyai.base.function_toolbox import custom_entropy,normalize,custom_novelty_calculation,spm_wnorm

A = np.array([[0.35,0.1,1.0],
              [0.3,0.8,0.0],
              [0.35,0.1,0.0]])
B = np.array([[0.35,0.1,1.0],
              [0.3,0.8,0.0],
              [1.35,1.1,1.0]])
# print()
# print("(Before):")
# before = (-custom_novelty_calculation(A))
# print("----------------------")
# print()
# print("(After):")
# after = (-custom_novelty_calculation(B))

def marginal_entropy(A,eps=1e-10):
    zeroes = np.zeros(A.shape)
    B = np.copy(A)
    B[A<eps] = eps
    entropy = -np.sum(A*np.log(B),0,keepdims=False)
    # O is an impossible value
    if (keepdims):
        return(zeroes + entropy)
    else : 
        return(entropy)

def marginal_entropy_gradient(A):
    # How much information gain would result in getting an update on a
    # specific state-observation coordinate ? 
    # H(A) - H(A|s)


before = custom_entropy(normalize(A))
after = custom_entropy(normalize(B))
ig = before - after # The information gain is the entropy of the previous model minus the
                    # the entropy of the updated model
print(before)
print(after)
print(ig)

# A = 10*A
# # print(custom_entropy(A,axis=0,keepdims=True))
# print()
# print("(New):")
# print(-custom_novelty_calculation(A))
# print("----------------------")
# print("(Old):")
# print(spm_wnorm(A))
# print()