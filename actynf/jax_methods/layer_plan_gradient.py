import jax.numpy as jnp
import jax.random as jr
import jax

from numpyro import plate,sample,deterministic
import numpyro.distributions as distr

from jax.tree_util import tree_map
from jax import vmap
from jax import jit

from functools import partial
from itertools import product

from .jax_toolbox import _normalize,_jaxlog
from .planning_tools import compute_Gt_array,compute_novelty
from .layer_infer_state import compute_state_posterior

# A set of functions for agents to plan their next moves.  Note that we use a bruteforce treesearch approach, which is
# a computational nightmare, but may be adapted to fit short term path planning.

# _________________________________________________________________________________________
# Functions used to build the policy tree. 
#  --> Add a level to the tree (changing a tensor of shape K to shape K x (Ph x Sh))
#  --> Evaluate the EFE at a specific level of the tree

# How would one do that ? 
# Precompute all action * (state - action)^Th paths. Prune the tree until only H remain
# Then scan and optimize (a bit like Policy gradients ?)





### Compute log policy posteriors --------------------------------------------------------------------------------
# @partial(jit, static_argnames=['Np','Th','gamma'])
def policy_posterior(current_timestep,Th,filter_end_of_trial,
                     qs,vecA,vecB,vecC,vecE,vecA_novel,vecB_novel,
                     planning_options):
    raise NotImplementedError("TODO !")
