import jax
import jax.numpy as jnp
import jax.random as jr

# From Eric Jang's blog
def sample_gumbel(shape, key,eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    u = jr.uniform(key, shape, minval=0, maxval=1)
    return -jnp.log(-jnp.log(u+eps)+eps)

def sample_gumbel_softmax(logits, temperature, key):
    """Sample from the Gumbel-Softmax distribution"""
    gumbel_noise = sample_gumbel(logits.shape, key)
    y = logits + gumbel_noise
    return jax.nn.softmax(y / temperature)

def gumbel_softmax(logits, temperature, rngkey,hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y = sample_gumbel_softmax(logits, temperature,rngkey)
    if hard:
        k = logits.shape[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = jax.nn.one_hot(jnp.argmax(y),k)
        y = jax.lax.stop_gradient(y_hard - y) + y
    return y

