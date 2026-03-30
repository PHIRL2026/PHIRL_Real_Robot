"""Train PHIRL on Kinova BrushEnv
Adapt PHIRL algorithm for real robot environment with manual reset between episodes.

Key differences from MiniGrid version:
- Single environment (no parallelization for real robot)
- 3D Cartesian observation space
- 6 discrete actions (no NO_OP during training)
- Manual reset with user confirmation between episodes
"""
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import rospy
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data import types
from imitation.rewards.reward_nets import RewardNet
from imitation.util import logger as imit_logger

# Import BrushEnv and wrappers
from env.BrushEnv import BrushEnv, StateSpace
from env.BrushEnvWithManualReset import BrushEnvWithManualReset

# Import reward networks
from training.DiscreteActionRewardNet import DiscreteActionRewardNet
from training.DiscreteActionShapedRewardNet import DiscreteActionShapedRewardNet

# Try to import PHIRL from the PHIRL_ICML repo
current_file = os.path.abspath(__file__)
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
PHIRL_ICML_REPO_PATH = os.path.join(grandparent_dir, "PHIRL_ICML")
sys.path.insert(0, PHIRL_ICML_REPO_PATH)

try:
    from IRL_lib_mod.phirl import PHIRL
    PHIRL_AVAILABLE = True
    print(f"Successfully imported PHIRL from {PHIRL_ICML_REPO_PATH}")
except ImportError as e:
    print(f"Warning: Could not import PHIRL from {PHIRL_ICML_REPO_PATH}")
    print(f"Error: {e}")
    PHIRL_AVAILABLE = False


@dataclass
class PHIRLBrushConfig:
    """Configuration for PHIRL training on BrushEnv"""
    # Environment
    max_steps: int = 100  # Max steps per episode
    
    # Network architecture (smaller for 3D obs space)
    reward_size: int = 32
    potential_size: int = 24
    
    # AIRL/PHIRL training
    n_disc: int = 2
    demo_batch_size: int = 2  # Smaller batch for fewer demos
    gen_buff_size: int = 256
    learning_rate: float = 3e-4
    demo_portion: float = 1.0
    
    # PPO parameters - single env settings
    ppo_batch_size: int = 8
    n_steps: int = 32  # Steps before PPO update
    n_epochs: int = 10
    gamma: float = 0.97
    gae_lambda: float = 0.95
    ent_coef: float = 0.01  # Lower entropy for real robot
    
    # PHIRL-specific
    shape_reward: List[str] = None  # ["reward_sign_loss", "advantage_sign_loss"]
    shaping_batch_size: int = 1
    shaping_update_freq: int = 1
    max_grad_norm: float = 0.5
    label_smoothing: float = 0.05
    
    # Training schedule
    training_rounds: int = 20
    steps_per_round: int = 1024  # Smaller for single env
    eval_episodes: int = 3  # Fewer eval episodes for real robot
    
    # Real robot settings
    auto_reset: bool = False  # Require manual reset
    
    def __post_init__(self):
        if self.shape_reward is None:
            self.shape_reward = ["reward_sign_loss", "advantage_sign_loss"]


def make_brush_env(config: PHIRLBrushConfig):
    """
    Create BrushEnv with manual reset wrapper.
    
    Returns a function that creates the environment (for DummyVecEnv compatibility).
    """
    def _init():
        env = BrushEnv(state_space=StateSpace.CARTESIAN, max_steps=config.max_steps)
        env = BrushEnvWithManualReset(env, auto_reset=config.auto_reset)
        return env
    return _init


def load_demonstrations(demo_dir: str, demo_portion: float = 1) -> Tuple[List[types.Trajectory], List[Dict], List[int]]:
    """
    Load annotated demonstrations in format compatible with PHIRL.
    
    Expected format in demo_dir/annotated/demo_X.json:
    {
        "observations": [[x, y, z], ...],
        "actions": [0, 1, 2, ...],
        "rewards": [0, 0, ..., 1],
        "dones": [false, ..., true],
        "annotations": [
            {"start_step": 0, "end_step": 10, "start_progress": 0.0, "end_progress": 20.0},
            ...
        ]
    }
    
    Returns:
        trajectories: List of imitation library Trajectory objects
        annotations: List of annotation dictionaries
        traj_indices: Index mapping for annotations
    """
    annotated_dir = os.path.join(demo_dir, "annotations")
    
    if not os.path.exists(annotated_dir):
        raise FileNotFoundError(f"Annotated demonstrations not found: {annotated_dir}")
    
    demo_files = sorted([f for f in os.listdir(annotated_dir) 
                        if f.startswith('demo_') and f.endswith('.json')
                        and 'summary' not in f],
                       key=lambda x: int(x.split('_')[1].split('.')[0]))

    if len(demo_files) > 0 and demo_portion < 1:
        num_to_sample = max(1, int(len(demo_files) * demo_portion))
        demo_files = sorted(
            random.sample(demo_files, num_to_sample),
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )

    if len(demo_files) == 0:
        raise FileNotFoundError(f"No demo files found in {annotated_dir}")
    
    trajectories = []
    all_annotations = []
    traj_indices = []
    
    for demo_file in demo_files:
        filepath = os.path.join(annotated_dir, demo_file)
        with open(filepath, 'r') as f:
            demo = json.load(f)
        
        # Convert to numpy arrays
        obs = np.array(demo['observations'], dtype=np.float32)
        acts = np.array(demo['actions'], dtype=np.int64)
        
        # Create Trajectory object
        # obs should have length len(acts) + 1 (observations before/after each action)
        if len(obs) == len(acts) + 1:
            traj_obs = obs
        elif len(obs) == len(acts):
            # Missing final observation - duplicate last one
            traj_obs = np.concatenate([obs, obs[-1:]], axis=0)
        else:
            # Too many observations - truncate
            traj_obs = obs[:len(acts) + 1]
        
        traj = types.Trajectory(
            obs=traj_obs,
            acts=acts,
            infos=None,
            terminal=True
        )
        trajectories.append(traj)
        
        # Process annotations
        demo_annotations = demo.get('annotations', [])
        for ann in demo_annotations:
            all_annotations.append(ann)
            traj_indices.append(len(trajectories) - 1)
    
    print(f"Loaded {len(trajectories)} trajectories with {len(all_annotations)} annotation segments")
    
    return trajectories, all_annotations, traj_indices


def train_phirl_brush(config: PHIRLBrushConfig, demo_dir: str, output_dir: str):
    """
    Train PHIRL on BrushEnv demonstrations.
    
    Args:
        config: Training configuration
        demo_dir: Directory containing annotated demonstrations
        output_dir: Directory to save trained models
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    if not PHIRL_AVAILABLE:
        raise ImportError(f"PHIRL is not available from {PHIRL_ICML_REPO_PATH}")
    
    print("="*80)
    print("PHIRL Training on Kinova BrushEnv")
    print("="*80)
    print(f"Max steps per episode: {config.max_steps}")
    print(f"Training rounds: {config.training_rounds}")
    print(f"Steps per round: {config.steps_per_round}")
    print(f"Shape reward losses: {config.shape_reward}")
    print(f"Auto reset: {config.auto_reset}")
    print()
    
    # Create single environment wrapped in DummyVecEnv for SB3 compatibility
    print("Creating BrushEnv...")
    venv = DummyVecEnv([make_brush_env(config)])
    
    print(f"Observation space: {venv.observation_space}")
    print(f"Action space: {venv.action_space}")
    
    # Load demonstrations
    print("\nLoading demonstrations...")
    demo_trajectories, _, _ = load_demonstrations(demo_dir, demo_portion=1.0)
    ann_trajectories, annotations, traj_indices = load_demonstrations(demo_dir, demo_portion=config.demo_portion)
    # Create annotation list in PHIRL format: list of (annotation_dict, traj_index)
    annotation_list = [(ann, traj_idx) for ann, traj_idx in zip(annotations, traj_indices)]
    
    # Initialize PPO learner
    print("\nInitializing PPO learner...")
    learner = PPO(
        policy="MlpPolicy",
        env=venv,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.ppo_batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=0.2,
        ent_coef=config.ent_coef,
        vf_coef=0.5,
        max_grad_norm=config.max_grad_norm,
        verbose=1
    )
    
    # Initialize reward network
    print("Initializing reward network...")
    reward_net = DiscreteActionShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        reward_hid_sizes=(config.reward_size, config.reward_size),
        potential_hid_sizes=(config.potential_size,),
        discount_factor=config.gamma,
    )
    
    # Initialize PHIRL trainer
    print("Initializing trainer...")
    
    custom_logger = imit_logger.configure(
        os.path.join(output_dir, "logs"),
        ["tensorboard", "stdout"]
    )
    
    trainer = PHIRL(
        demonstrations=demo_trajectories,
        demo_batch_size=config.demo_batch_size,
        gen_replay_buffer_capacity=config.gen_buff_size,
        n_disc_updates_per_round=config.n_disc,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        shape_reward=config.shape_reward,
        annotation_list=annotation_list,
        demostrations_for_shaping=ann_trajectories,
        custom_logger=custom_logger,
        save_path=output_dir,
        shaping_batch_size=config.shaping_batch_size,
        traj_index=traj_indices,
        allow_variable_horizon=True,
        save_model_every=100,
        shaping_update_freq=config.shaping_update_freq,
        label_smoothing_eps=config.label_smoothing,
    )
    print("Using PHIRL with progress shaping")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("NOTE: You will be prompted to manually reset the environment between episodes.")
    print("="*80 + "\n")
    
    rewards_history = []
    success_history = []
    
    for round_idx in range(config.training_rounds):
        print(f"\n{'='*40}")
        print(f"Round {round_idx + 1}/{config.training_rounds}")
        print(f"{'='*40}")
        
        # Train for one round
        trainer.train(total_timesteps=config.steps_per_round)
        
        # Evaluate (with manual resets)
        print("\nEvaluation phase...")
        eval_rewards = []
        eval_successes = 0
        
        for eval_ep in range(config.eval_episodes):
            print(f"\nEvaluation episode {eval_ep + 1}/{config.eval_episodes}")
            
            obs = venv.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done and steps < config.max_steps:
                action, _ = learner.predict(obs, deterministic=True)
                obs, reward, done, info = venv.step(action)
                episode_reward += reward[0]
                steps += 1
                done = done[0]
            
            eval_rewards.append(episode_reward)
            if episode_reward > 0:
                eval_successes += 1
            
            print(f"  Episode reward: {episode_reward:.3f}, Steps: {steps}")
        
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        rewards_history.append(mean_reward)
        success_history.append(eval_successes)
        
        print(f"\nRound {round_idx + 1} Summary:")
        print(f"  Mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")
        print(f"  Success rate: {eval_successes}/{config.eval_episodes}")
        
        # Save checkpoint every 5 rounds
        if (round_idx + 1) % 5 == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_round_{round_idx + 1}")
            learner.save(checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(output_dir, "phirl_brush_final")
    learner.save(final_path)
    print(f"\nFinal model saved: {final_path}")
    
    # Save reward network separately
    reward_net_path = os.path.join(output_dir, "reward_net.pt")
    th.save(trainer.reward_train.state_dict(), reward_net_path)
    print(f"Reward network saved: {reward_net_path}")
    
    # Save training statistics
    stats = {
        'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
        'rewards_history': [float(r) for r in rewards_history],
        'success_history': success_history,
        'final_mean_reward': float(rewards_history[-1]) if rewards_history else 0,
        'best_reward': float(max(rewards_history)) if rewards_history else 0,
        'best_success_rate': max(success_history) / config.eval_episodes if success_history else 0
    }
    
    stats_path = os.path.join(output_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best reward: {stats['best_reward']:.3f}")
    print(f"Best success rate: {stats['best_success_rate']:.1%}")
    print("="*80)
    
    venv.close()
    
    return stats


if __name__ == "__main__":
    try:
        rospy.init_node('train_phirl_brush', anonymous=True)
    except:
        raise Exception("ROS node initialization failed")

    # Get the directory where this script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    
    # Configuration
    config = PHIRLBrushConfig(
        max_steps=64,
        training_rounds=50,
        steps_per_round=256,
        shape_reward=["reward_sign_loss", "advantage_sign_loss"],  # Enable PHIRL shaping
        auto_reset=False,  # Require manual reset
        eval_episodes=1,
        demo_portion= 0.5, 
    )

    # Experiment Name
    experiment_name = "EXP_1"
    
    # Paths
    demo_dir = os.path.join(PROJECT_DIR, "demonstrations")

    # Save model checkpoints and logs to output_dir. Create folder if not exist
    output_dir = os.path.join(PROJECT_DIR, "experiments", experiment_name, "phirl_trained")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if demonstrations exist
    annotated_dir = os.path.join(demo_dir, "annotations")
    if not os.path.exists(annotated_dir):
        print(f"Error: Annotated demonstrations not found in {annotated_dir}")
        print("\nPlease ensure you have:")
        print("1. Recorded demonstrations using demo_recorder.py")
        print("2. Annotated them with progress labels")
        print("3. Saved them in demonstrations/annotations/demo_X.json format")
        print("\nExpected format:")
        print('{')
        print('  "observations": [[x, y, z], ...],')
        print('  "actions": [0, 1, 2, ...],')
        print('  "annotations": [')
        print('    {"start_step": 0, "end_step": 10, "start_progress": 0.0, "end_progress": 20.0},')
        print('    ...')
        print('  ]')
        print('}')
        exit(1)
    
    # Train PHIRL
    train_phirl_brush(config, demo_dir, output_dir)
