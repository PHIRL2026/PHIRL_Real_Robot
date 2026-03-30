from enum import IntEnum
import numpy as np

"""
Discretized actions. We have 7 actions, one for each x,y,z axis in either direction and no op. 

"""
class DiscreteActions(IntEnum):
    MOVE_X = 0 # left 
    MOVE_Y = 1 # back 
    MOVE_Z = 2 # up
    MOVE_X_NEG = 3 # right
    MOVE_Y_NEG = 4 # forward
    MOVE_Z_NEG = 5  # down 
    NO_OP = 6

    def to_cartesian_array(self):
        """Convert action to a cartesian pose array: [x, y, z, theta_x, theta_y, theta_z]"""
        if self == DiscreteActions.MOVE_X:
            return [1, 0, 0, 0, 0, 0]
        elif self == DiscreteActions.MOVE_X_NEG:
            return [-1, 0, 0, 0, 0, 0]
        elif self == DiscreteActions.MOVE_Y:
            return [0, 1, 0, 0, 0, 0]
        elif self == DiscreteActions.MOVE_Y_NEG:
            return [0, -1, 0, 0, 0, 0]
        elif self == DiscreteActions.MOVE_Z:
            return [0, 0, 1, 0, 0, 0]
        elif self == DiscreteActions.MOVE_Z_NEG:
            return [0, 0, -1, 0, 0, 0]
        elif self == DiscreteActions.NO_OP:
            return [0, 0, 0, 0, 0, 0]
        else:
            raise ValueError(f"Unknown action {self}")

"""
This class wraps the joy commands and button inputs into discrete actions.
"""
class DiscreteActionWrapper:
    def __init__(self):
        pass

    def get_action(self, commands, buttons):
        action = None

        # Create a vector, find vector with largest magnitude. We want to convert this 3D command vector into one discrete action.  
        vec = np.array([commands.linear.x,
                        commands.linear.y,
                        commands.linear.z])

        idx = np.argmax(np.abs(vec))

        # x
        if idx == 0 and vec[idx] > 0:
            action = DiscreteActions.MOVE_X
        elif idx == 0:
            action = DiscreteActions.MOVE_X_NEG
        
        # y
        if idx == 1 and vec[idx] > 0:
            action = DiscreteActions.MOVE_Y_NEG
        elif idx == 1:
            action = DiscreteActions.MOVE_Y

        # z 
        if idx == 2 and vec[idx] > 0:
            action = DiscreteActions.MOVE_Z
        elif idx == 2:
            action = DiscreteActions.MOVE_Z_NEG

        # no op if value too small
        if np.isclose(vec[idx], 0.0):
            action = DiscreteActions.NO_OP


        return action