from .memory_decay import MemoryDecayType
from .space_structure import AssumedSpaceStructure

# Assign enums to variables for direct access
NO_STRUCTURE = AssumedSpaceStructure.NO_STRUCTURE
LINEAR = AssumedSpaceStructure.LINEAR
LINEAR_CLAMPED = AssumedSpaceStructure.LINEAR_CLAMPED
LINEAR_PERIODIC = AssumedSpaceStructure.LINEAR_PERIODIC

NO_MEMORY_DECAY = MemoryDecayType.NO_MEMORY_DECAY
PROPORTIONAL_MEMORY_DECAY = MemoryDecayType.PROPORTIONAL
STATIC_MEMORY_DECAY= MemoryDecayType.STATIC

__all__ = ['MemoryDecayType', 'AssumedSpaceStructure', 
           'NO_STRUCTURE', 'LINEAR', 'LINEAR_CLAMPED', 'LINEAR_PERIODIC', 
           'NO_MEMORY_DECAY', 'PROPORTIONAL_MEMORY_DECAY','STATIC_MEMORY_DECAY']