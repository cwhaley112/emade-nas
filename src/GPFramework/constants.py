"""
Coded by Joel Ye and Austin Dunn
Constants
"""
import enum

class TriState(enum.Enum):
    """
    Class which can take on three possible values.
    Used to specify how to load and store an instance of data
    """
    FEATURES_TO_FEATURES = 0
    STREAM_TO_STREAM = 1
    STREAM_TO_FEATURES = 2

class Axis(enum.Enum):
    """
    Class which can take on four possible values.
    Used to specify which axis to evaluate the data on 
    Or whether to send the full set of data to evaluation
    """
    AXIS_0 = 0
    AXIS_1 = 1
    AXIS_2 = 2
    FULL = 3

# Helpful constants
TRI_STATE = [TriState.FEATURES_TO_FEATURES, TriState.STREAM_TO_STREAM, TriState.STREAM_TO_FEATURES]
AXIS = [Axis.AXIS_0, Axis.AXIS_1, Axis.AXIS_2, Axis.FULL]
FEATURES_TO_FEATURES = TriState.FEATURES_TO_FEATURES
STREAM_TO_STREAM = TriState.STREAM_TO_STREAM
STREAM_TO_FEATURES = TriState.STREAM_TO_FEATURES

# sets __repr__ to point to __str__
# deap uses __repr__ which will not evaluate correctly unless __str__ is used instead
TriState.__repr__ = TriState.__str__
Axis.__repr__ = Axis.__str__
TIERED_MULTIPLIER = 5e9
