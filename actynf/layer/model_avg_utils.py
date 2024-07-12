
import numpy as np

from ..base.miscellaneous_toolbox import isField,listify,flexible_copy
from ..base.function_toolbox import normalize, spm_kron,spm_dekron

def joint_to_kronecker(joint_distribution):
    """ Warning ! For a single timestep."""
    return joint_distribution.flatten('C')

def joint_to_kronecker_accross_time(joint_distribution):
    timedim = joint_distribution.shape[-1]
    return np.reshape(joint_distribution,(-1,timedim),order='C')
    # return joint_distribution.reshape(-1, timedim,order='C') 

def kronecker_to_joint(flat_distribution,Ns):
    return np.reshape(flat_distribution,Ns,'C')

def kronecker_to_joint_accross_time(x_kron,Ns):
    return np.reshape(x_kron,Ns + [-1],'C')

def dekron_state(x_kron,Ns,at_time=0):
    # assert at_time<self.T,"Can't get kronecker form of hidden states at time " + at_time + " (Temporal horizon reached)"
    return  spm_dekron(x_kron[at_time],tuple(Ns))

# Utilitaries
def factorwise_action_model_average(B,U, action_distribution):
    Nf = len(B)
    Np = action_distribution.shape[0]
    
    return_this_matrix_list = []
    for factor in range(Nf):
        sum_of_matrices = 0
        for policy in range(Np):
            action_K_done = U[policy,factor]
            prob_action_K = action_distribution[policy]
            sum_of_matrices += prob_action_K*B[factor][:,:,action_K_done]
        return_this_matrix_list.append(normalize(sum_of_matrices))
    return return_this_matrix_list

def kronecker_action_model_average(b_kron, action_distribution, just_slice=False):
    if (just_slice):
        action_id = action_distribution
        return b_kron[action_id]
    else:
        kron_b_arr = np.array(b_kron)
        return np.einsum("iju,u->ij",b_kron,action_distribution)
        # return np.average(kron_b_arr,axis=0,weights=action_distribution)

def get_factorwise_actions(self,at_time=0):
    assert at_time<self.T-1,"Can't get kronecker form of hidden states at time " + at_time + " (Temporal horizon reached)"
    return (self.U[self.STM.u[at_time],:]).tolist()

def get_kron_state_at_time(self,at_time=0):
    assert at_time<self.T,"Can't get kronecker form of hidden states at time " + at_time + " (Temporal horizon reached)"
    if (self.STM.is_value_exists("x_d",at_time)):
        return self.joint_to_kronecker(self.STM.x_d[...,at_time])
    else : 
        return spm_kron(self.var.d)

def get_total_number_of_hidden_states(Ns):
    return np.prod(Ns)
