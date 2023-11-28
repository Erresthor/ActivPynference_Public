from enum import Enum

# MEMORY DECAY MECHANICS : 
class MemoryDecayType(Enum):
    NO_MEMORY_DECAY = 0
    PROPORTIONAL = 1
    STATIC = 2