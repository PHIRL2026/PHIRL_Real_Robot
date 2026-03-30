from typing import Tuple

import gymnasium as gym
import torch as th
import torch.nn as nn
from imitation.rewards.reward_nets import RewardNet


class DiscreteActionRewardNet(RewardNet):
    """
    Reward network for discrete actions with embedding.
    
    Adapted for BrushEnv's 3D Cartesian observation space.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = True,
    ):
        super().__init__(observation_space, action_space)
        
        self.use_state = use_state
        self.use_action = use_action
        self.use_next_state = use_next_state
        
        # Get observation dimension (3 for BrushEnv Cartesian)
        obs_dim = observation_space.shape[0]
        
        # Handle discrete action space
        if isinstance(action_space, gym.spaces.Discrete):
            self.discrete_actions = True
            self.n_actions = action_space.n
            # Use embedding for discrete actions
            self.action_embedding = nn.Embedding(self.n_actions, 8)  # Smaller embedding for 6 actions
            action_dim = 8
        else:
            self.discrete_actions = False
            action_dim = action_space.shape[0]
            self.action_embedding = None
        
        # Calculate input dimension
        input_dim = 0
        if use_state:
            input_dim += obs_dim
        if use_action:
            input_dim += action_dim
        if use_next_state:
            input_dim += obs_dim
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.ReLU(),
            ])
            prev_dim = hidden_size
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Forward pass of reward network."""
        inputs = []
        batch_size = state.shape[0]
        
        if self.use_state:
            inputs.append(state.float())
        
        if self.use_action:
            if self.discrete_actions:
                # Handle discrete actions - convert to embedding
                action_long = action.long()
                
                if action_long.numel() > batch_size:
                    action_long = action_long.flatten()[:batch_size]
                elif action_long.dim() == 0:
                    action_long = action_long.unsqueeze(0)
                elif action_long.dim() == 2:
                    action_long = action_long.squeeze(-1)
                    
                action_long = action_long.view(batch_size)
                action_emb = self.action_embedding(action_long)
                inputs.append(action_emb)
            else:
                inputs.append(action.float())
        
        if self.use_next_state:
            inputs.append(next_state.float())
        
        x = th.cat(inputs, dim=-1)
        return self.mlp(x).squeeze(-1)