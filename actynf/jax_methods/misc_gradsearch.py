import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import jax
from jax.tree_util import tree_map
from jax import lax,vmap,jit

from jax_toolbox import _normalize
from planning_tools import compute_Gt_array,compute_novelty

import tensorflow_probability.substrates.jax.distributions as tfd

def compute_G_pi(qpi,qs_tminus,
                 A,B,C,
                 A_novel,B_novel):
    # At a given timestep t_exp = t + i, with i in [0,Th[
    # We explore a given Qpi, distribution over the chosen policy

    # Qpi_t = array
    B_pi_t = jnp.einsum("ijk,k->ij",B,qpi)
    qs_pi_tplus,_ = _normalize(B_pi_t @ qs_tminus) # Useless norm ?

    # qo is the list of predicted observation distributions at this time t given this qs_pi_tplus !
    qo = tree_map(lambda a_m : a_m @ qs_pi_tplus,A)

    Gt = compute_Gt_array(qo,qs_pi_tplus,qs_tminus,qpi,
                          A,A_novel,
                          B,B_novel,C)
    return Gt,qs_pi_tplus

def future_search(qpi_allts,
                  qs_init,
                A,B,C,E,
                A_novel,B_novel):
    """
    Performing gradient descent on sum(complete_G_array) 
    -> Agent planning without relying on a policy tree search ! 
    Good scaling on the Th dimension, maybe less efficient when trying to find 
    narrow paths, may be further optimized with :
     - A discount on further outcomes
     - A dynamic gradient through iterations ?
    """

    Th = qpi_allts.shape[-1]

    def _scanner(carry,t):
        qs = carry

        qpi = qpi_allts[:,t]

        Gt,qs_next = compute_G_pi(qpi,qs,
                                    A,B,C,
                                    A_novel,B_novel)

        return qs_next,(qs,Gt)

    qs = qs_init
    # G_arr = jnp.empty((4,))
    qs_horizon,(qss,complete_G_array) = jax.lax.scan(_scanner, qs, jnp.arange(0,Th,1))

    return qss,complete_G_array

def policy_gradient_selection(initial_qpi_allts,qsm,
                            A,B,C,E,A_novel,B_novel,
                            lr = 0.1,
                            verbose=True):
    
    Gpi =  (lambda x : future_search(x,qsm,A,B,C,E,A_novel,B_novel))
        # yields qss,Gpi
    
    loss = (lambda x : -Gpi(x)[1].sum())
        # yields \sum{Gpi}
    d_loss = jax.grad(loss)

    losses = []
    qpi = initial_qpi_allts
    
    old_loss = 10000
    slider_max_len = 16

    eps = 1e0
    loss_history_window = []
    last_loss_improvement = 10000
    while ((last_loss_improvement>eps)) or (len(loss_history_window)<slider_max_len):
        Xqpi = jax.nn.softmax(qpi,axis=0)  # Encode the parameter vector

        new_loss = loss(Xqpi)
        grad = d_loss(Xqpi)

        qss,Gpis = Gpi(Xqpi)

        if verbose :
            print("LOSS : " + str(new_loss))
        losses.append(new_loss)

        loss_history_window.append(new_loss)
        if len(loss_history_window)>slider_max_len:
            loss_history_window.pop(0)
        last_loss_improvement = new_loss - min(loss_history_window)
        

        old_loss=new_loss
        qpi = qpi - lr*grad

    vector_posterior = qpi
    action_posterior = (jax.nn.softmax(vector_posterior,axis=0))
    return action_posterior,qss,qpi,losses

if __name__=="__main__":
    import random as ra
    import numpy as np
    Nos = np.array([10,3,2])
    Ns = 10
    T = 10
    Np = 10

    key = jr.PRNGKey(46463)

    fixed_observations = [np.random.randint(0,No,(T,)) for No in Nos]
    obs_vectors = [jax.nn.one_hot(rvs,No,axis=0) for rvs,No in zip(fixed_observations,Nos)]

    # A = [_normalize(jr.uniform(key,(No,Ns)))[0] for No in Nos]

    Nmod = 2
    A = [_normalize(jnp.eye(Ns))[0] for i in range(Nmod)]
    

    C = [jnp.zeros((a.shape[0],)) for a in A]
    C[1] = jnp.linspace(0,10,C[1].shape[0])

    obs_vectors = [_normalize(jr.uniform(key,(No,)))[0] for No in Nos]

    B = np.zeros((Ns,Ns,Np))
    for u in range(Ns):
        B[:,:,u] = np.eye(Ns)
        try :
            B[u+1,u,u] = 0.5
            B[u,u,u] = 0.5
        except:
            continue
    
    
    
    A_novel = compute_novelty(A,True)
    B_novel = compute_novelty(B)

    E = jnp.ones((Np,))


    # Planning next action for the next Th tmstps !
    Th = 25    # qpi_allts,_ = _normalize(jr.uniform(key,(Np,Th)))

    key,lockey = jr.split(key)
    qsm,_ = _normalize(jr.uniform(lockey,(Ns,),minval=0,maxval=9))
    qsm = jax.nn.one_hot(5,Ns)
    key,lockey = jr.split(key)
    qsp,_ = _normalize(jr.uniform(lockey,(Ns,),minval=0,maxval=9))    
    qpi_allts,_ = _normalize(jnp.ones((Np,Th)))


    action_posterior,qss,qpi,losses = policy_gradient_selection(qpi_allts,qsm,
                                    A,B,C,E,A_novel,B_novel,lr=0.25)

    fig,axs = plt.subplots(2,1)
    ax = axs[0]
    ax.imshow(action_posterior,vmin=0,vmax=1)
    ax.set_title("Actions posterior")

    ax=axs[1]
    ax.imshow(jnp.swapaxes(qss,0,1),vmin=0,vmax=1)
    ax.set_title("States posterior")
    fig.show()

    input()