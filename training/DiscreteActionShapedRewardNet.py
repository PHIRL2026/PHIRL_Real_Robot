from typing import Tuple

import gymnasium as gym
import torch as th
import torch.nn as nn
from imitation.rewards.reward_nets import RewardNet

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.DiscreteActionRewardNet import DiscreteActionRewardNet


class DiscreteActionShapedRewardNet(RewardNet):
    """
    Shaped reward network for discrete actions.
    
    Combines a base reward network with a potential-based shaping function.
    Compatible with PHIRL's ShapedRewardNet interface.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        reward_hid_sizes: Tuple[int, ...] = (64, 64),
        potential_hid_sizes: Tuple[int, ...] = (32,),
        discount_factor: float = 0.99,
    ):
        super().__init__(observation_space, action_space)
        
        self.discount_factor = discount_factor
        
        # Base reward network
        self._base = DiscreteActionRewardNet(
            observation_space=observation_space,
            action_space=action_space,
            hidden_sizes=reward_hid_sizes,
            use_state=True,
            use_action=True,
            use_next_state=True,
        )
        
        # Potential network (state-only)
        obs_dim = observation_space.shape[0]
        pot_layers = []
        prev_dim = obs_dim
        for hidden_size in potential_hid_sizes:
            pot_layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.ReLU(),
            ])
            prev_dim = hidden_size
        pot_layers.append(nn.Linear(prev_dim, 1))
        self._potential = nn.Sequential(*pot_layers)
    
    @property
    def base(self) -> DiscreteActionRewardNet:
        """Returns the base reward network."""
        return self._base
    
    def potential(self, state: th.Tensor) -> th.Tensor:
        """Compute potential value for state."""
        return self._potential(state.float())
    
    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """
        Compute shaped reward: R(s,a,s') + gamma * Phi(s') - Phi(s)
        """
        base_reward = self._base(state, action, next_state, done)
        
        # Potential-based shaping
        pot_s = self.potential(state).squeeze(-1)
        pot_s_next = self.potential(next_state).squeeze(-1)
        
        # Shaped reward
        shaped_reward = base_reward + self.discount_factor * pot_s_next - pot_s
        
        return shaped_reward