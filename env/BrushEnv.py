from typing import Optional
import numpy as np
import gymnasium as gym
from enum import Enum
import sys
import os
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.phirl_arm import Arm
from env.BrushPerception import BrushPerception
from demonstrations.DiscreteActionWrapper import DiscreteActions

# scale and duration of the velocity vector. 
SCALE = 0.1
DURATION = 0.25

class StateSpace(Enum):
    CARTESIAN = 1,
    JOINT = 2
    IMAGE = 3
    CARTESIAN_IMAGE = 4

class BrushEnv(gym.Env):

    def __init__(self, state_space=StateSpace.CARTESIAN, max_steps=100, image_height=480, image_width=640):
        self.arm = Arm()
        self.perception = BrushPerception()
        self.action_space = gym.spaces.Discrete(6)

        self.state_space = state_space
        self.max_steps = max_steps
        self.steps = 0
        self.image_height = image_height
        self.image_width = image_width

        cartesian_mins = [0.25, 0.1, 0.14]
        cartesian_maxs = [0.55, 0.34, 0.45]

        self._cartesian_bounds = gym.spaces.Box(
            low=np.array(cartesian_mins, dtype=np.float32),
            high=np.array(cartesian_maxs, dtype=np.float32),
            dtype=np.float32
        )

        if state_space == StateSpace.CARTESIAN:
            self.observation_space = gym.spaces.Box(
                low=np.array(cartesian_mins, dtype=np.float32),
                high=np.array(cartesian_maxs, dtype=np.float32),
                dtype=np.float32
            )
        elif state_space == StateSpace.IMAGE:
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(image_height, image_width, 3),
                dtype=np.uint8
            )
        elif state_space == StateSpace.CARTESIAN_IMAGE:
            self.observation_space = gym.spaces.Dict({
                'cartesian': gym.spaces.Box(
                    low=np.array(cartesian_mins, dtype=np.float32),
                    high=np.array(cartesian_maxs, dtype=np.float32),
                    dtype=np.float32
                ),
                'image': gym.spaces.Box(
                    low=0, high=255,
                    shape=(image_height, image_width, 3),
                    dtype=np.uint8
                )
            })
        else:
            raise NotImplementedError("Joint Space currently not supported")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        super().reset(seed=seed)

        self.arm.clear_faults()
        self.arm.home_arm()
        self.steps = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action, move=True):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """ 

        if self._valid_action(action) and move:
            self._move_arm(action)
        else:
            print("Action goes out of bounds!")

        # check terminal condition
        terminated = self._is_terminal()

        # check truncate condition
        truncated = self.steps > self.max_steps

        # reward
        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    def render(self):
        raise Exception("This environment does not support rendering")
    

    def close(self):
        # do we need any clean up?
        return
    
    def _is_terminal(self):
        return self.perception.count_knot_pixels() < 150

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: either cartesian state or joint state
        """
        if self.state_space == StateSpace.CARTESIAN:
            cartesian_pose = self.arm.get_cartesian_pose()

            return cartesian_pose[:3]
        elif self.state_space == StateSpace.JOINT:
            return self.arm.get_joint_pose()
        elif self.state_space == StateSpace.IMAGE:
            img = self.perception.get_latest_image()
            if img is None:
                return np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            return cv2.resize(img, (self.image_width, self.image_height))
        elif self.state_space == StateSpace.CARTESIAN_IMAGE:
            cartesian_pose = self.arm.get_cartesian_pose()
            img = self.perception.get_latest_image()
            if img is None:
                img = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (self.image_width, self.image_height))
            return {
                'cartesian': np.array(cartesian_pose[:3], dtype=np.float32),
                'image': img
            }
        else:
            raise Exception() 
    
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info 
        """
        return {}
    
    def _move_arm(self, action):
        try:
            velocity = np.array(DiscreteActions(action).to_cartesian_array()) * 0.1

            self.arm.cartesian_velocity_command(
                velocity,
                duration=0.25,
                radians=True
            )
        except Exception as e:
            print(f"Failed to step: {e}")
            return False

        return True
    
    def _valid_action(self, action):
        cartesian_pose = self.arm.get_cartesian_pose()
        curr_obs = np.asarray(cartesian_pose[:3], dtype=np.float32)
        action_dist = SCALE * DURATION

        delta = np.zeros(3, dtype=np.float32)

        if action == DiscreteActions.MOVE_X:
            delta[0] = action_dist
        elif action == DiscreteActions.MOVE_X_NEG:
            delta[0] = -action_dist
        elif action == DiscreteActions.MOVE_Y:
            delta[1] = action_dist
        elif action == DiscreteActions.MOVE_Y_NEG:
            delta[1] = -action_dist
        elif action == DiscreteActions.MOVE_Z:
            delta[2] = action_dist
        else:
            delta[2] = -action_dist

        next_obs = curr_obs + delta
        next_obs = next_obs.astype(self._cartesian_bounds.dtype)

        return self._cartesian_bounds.contains(next_obs)

        
    
