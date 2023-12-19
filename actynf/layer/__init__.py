from .model_layer import mdp_layer
from .layer_components import link_function
# Need to add mdp_layer, link and network to that !

layer = mdp_layer
link = link_function

__all__ = ['layer','link']