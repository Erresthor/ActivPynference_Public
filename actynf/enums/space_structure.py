from enum import Enum

class AssumedSpaceStructure(Enum):
    NO_STRUCTURE = 0
    LINEAR = 1 # Indices are related to a "spatial" position. Distance between two indices is 
                # proportional to distance between positions.
                # We  will not be interpolating for a transition to states outside the definition interval
    LINEAR_CLAMPED = 2
                # Indices are related to a "spatial" position. Distance between two indices is 
                # proportional to distance between positions. 
                # When interpolating for a transition to states outside the definition interval,
                # we will assume that we are clamped to the maximum / minimum possible value
    LINEAR_PERIODIC = 3 
                # Indices are related to a "spatial" position. Distance between two indices is 
                # proportional to distance between positions. 
                # When interpolating for a transition to states outside the definition interval,
                # we will assume that the space is periodic (going beyond the upper bound leads to 
                # a return to lower bound values . e.g : angles)
