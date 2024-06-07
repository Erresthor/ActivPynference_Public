__version__ = '0.1.33'
__author__ = 'Come Annicchiarico'
__credits__ = 'Centre de Recherche en Neurosciences de Lyon'

from .enums import MemoryDecayType,AssumedSpaceStructure
from .enums import NO_MEMORY_DECAY,PROPORTIONAL_MEMORY_DECAY,STATIC_MEMORY_DECAY
from .enums import NO_STRUCTURE,LINEAR,LINEAR_CLAMPED,LINEAR_PERIODIC

from .base import normalize,softmax,isField

from .layer import layer,link
from .architecture import layer_network
# Need to add mdp_layer, link and network to that !

__all__ = ['MemoryDecayType', 'AssumedSpaceStructure', 
           'NO_STRUCTURE', 'LINEAR', 'LINEAR_CLAMPED', 'LINEAR_PERIODIC', 
           'NO_MEMORY_DECAY', 'PROPORTIONAL_MEMORY_DECAY','STATIC_MEMORY_DECAY',
           'normalize','softmax','isField',
           'layer','link','layer_network']