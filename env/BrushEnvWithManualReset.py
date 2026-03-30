import gymnasium as gym

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.BrushEnv import BrushEnv


class BrushEnvWithManualReset(gym.Wrapper):
    """
    Wrapper that pauses for user input before resetting the environment.
    This allows the human operator to physically reset the robot/task.
    """
    
    def __init__(self, env: BrushEnv, auto_reset: bool = False):
        super().__init__(env)
        self.auto_reset = auto_reset
        self.episode_count = 0
    
    def reset(self, **kwargs):
        if not self.auto_reset and self.episode_count > 0:
            print("\n" + "="*60)
            print("EPISODE ENDED - Manual Reset Required")
            print("="*60)
            print("Please reset the physical environment (brush position, etc.)")
            input("Press ENTER when ready to continue training...")
            print("Resuming training...\n")
        
        self.episode_count += 1
        return self.env.reset(**kwargs)