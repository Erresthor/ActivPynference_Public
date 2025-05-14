from actynf.jaxtynf.jax_toolbox import spm_wnorm,_normalize
import jax
import jax.numpy as jnp


if __name__ == "__main__":
    
    
    qs = jnp.array([0.5,0.5])
    # If the predicted state distribution has high entropy (e.g. [0.5,0.5]) 
    # Thus result in a poorly defined predicted observation dist ([0.5,0.5]),  
    # for a quite well defined likelihood mapping :
    # A = [  0.01  0.99]
    #     [  0.99  0.01]
    # This results in very high novelty counts whereas the subjct's model may be well fit,
    # resulting in degenrating behaviour ...
    # W = [ -10.0 -0.01]
    #     [ -0.01 -10.0]
    # This is probably a result of a MFA somewhere ? 
    # Here's a hack to avoid this kind of situation :
    K =100
    pA = jnp.array([
        [0.01,0.99*K],
        [0.99*K,0.01]
    ])
    W = spm_wnorm(pA)
    A ,_=_normalize(pA)
    
    qo = jnp.einsum("ij,j->i",A,qs)  # Predicted observation, $o_t = As_t$
    old_novelty = jnp.einsum("os,o,s->",W,qo,qs)
    print(old_novelty)  # Significantly high, tricking the agent into favoring this outcome
                        # That can be excused if the agent does not face too much uncertainty 
                        # when planning actions. But the agent will likely never solve this 
                        # if it can't completely remove state uncertainty due to noisy priors...
    
    
    
    # For each possible unit state s, the corresponding observation distribution is given by the matrix A
    # in the s-th column : A[i,j=s]
    # Each of those distributions has an individual associated novelty encoded in W
    # new_novelty = 
    novelty_depending_on_state = (A*W).sum(axis=0)

    print(novelty_depending_on_state)
    
    expected_novelty = jnp.einsum("i,j->",novelty_depending_on_state,qs)
    
    print(expected_novelty)  # Significantly lower than the old one, 
                             # The agent is aware that emission that would seem "novel"
                             # in the old formulation are in fact caused by well-known dynamics