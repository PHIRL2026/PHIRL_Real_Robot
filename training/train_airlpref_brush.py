import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch as th
import rospy
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import types
from imitation.util import logger as imit_logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.BrushEnv import BrushEnv, StateSpace
from env.BrushEnvWithManualReset import BrushEnvWithManualReset
from training.DiscreteActionRewardNet import DiscreteActionRewardNet
from training.train_pref_brush import (
    build_fragment_records,
    build_preference_dataset,
    get_annotation_files,
    get_brush_spaces,
    load_policy_dependencies,
    load_preference_dependencies,
    normalize_demo_arrays,
)


@dataclass
class AIRLPrefBrushConfig:
    max_steps: int = 64
    reward_size: int = 32
    demo_portion: float = 1.0
    seed: int = 0
    auto_reset: bool = False
    learning_rate: float = 3e-4
    ppo_batch_size: int = 8
    n_steps: int = 32
    n_epochs: int = 10
    gamma: float = 0.97
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_disc: int = 2
    demo_batch_size: int = 2
    gen_buff_size: int = 256
    airl_training_rounds: int = 50
    airl_steps_per_round: int = 256
    airl_eval_episodes: int = 1
    tie_threshold: float = 2.0
    max_preferences: Optional[int] = 1024
    comparison_queue_size: Optional[int] = None
    min_fragment_steps: int = 1
    preference_noise_prob: float = 0.0
    preference_discount_factor: float = 1.0
    preference_model_threshold: float = 50.0
    reward_batch_size: int = 8
    reward_minibatch_size: Optional[int] = None
    reward_epochs: int = 20
    reward_learning_rate: float = 1e-3
    run_policy_stage: bool = True
    continue_policy_from_airl: bool = True
    policy_training_rounds: int = 20
    policy_steps_per_round: int = 256
    policy_eval_episodes: int = 1
    experiment_name: str = "EXP_1"


def make_brush_env(config: AIRLPrefBrushConfig):
    def _init():
        env = BrushEnv(state_space=StateSpace.CARTESIAN, max_steps=config.max_steps)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_steps)
        env = BrushEnvWithManualReset(env, auto_reset=config.auto_reset)
        return env
    return _init


def ensure_ros_node(node_name: str) -> None:
    if not rospy.core.is_initialized():
        rospy.init_node(node_name, anonymous=True)


def prompt_for_stage_start(stage_name: str, auto_reset: bool) -> None:
    if not auto_reset:
        print("\n" + "=" * 60)
        print(f"Prepare physical setup for {stage_name}")
        print("=" * 60)
        input("Press ENTER to continue...")


def write_json(file_path: str, content: Dict[str, Any]) -> None:
    with open(file_path, "w") as file_handle:
        json.dump(content, file_handle, indent=2)


def build_reward_net(config: AIRLPrefBrushConfig) -> DiscreteActionRewardNet:
    observation_space, action_space = get_brush_spaces()
    return DiscreteActionRewardNet(
        observation_space=observation_space,
        action_space=action_space,
        hidden_sizes=(config.reward_size, config.reward_size),
        use_state=True,
        use_action=True,
        use_next_state=True,
    )


def load_selected_demonstrations(file_paths: Sequence[str]):
    airl_trajectories: List[types.Trajectory] = []
    preference_trajectories: List[Any] = []
    annotations_by_traj: List[Tuple[str, Sequence[Dict[str, Any]]]] = []
    for file_path in file_paths:
        with open(file_path, "r") as file_handle:
            demo = json.load(file_handle)
        obs, acts, rews, dones = normalize_demo_arrays(demo)
        terminal = bool(dones[-1]) if len(dones) else True
        infos = np.array([{} for _ in range(len(acts))], dtype=object)
        airl_trajectories.append(
            types.Trajectory(
                obs=obs.copy(),
                acts=acts.copy(),
                infos=None,
                terminal=terminal,
            )
        )
        preference_trajectories.append(
            types.TrajectoryWithRew(
                obs=obs.copy(),
                acts=acts.copy(),
                infos=infos,
                terminal=terminal,
                rews=rews.copy(),
            )
        )
        annotations_by_traj.append((os.path.basename(file_path), demo.get("annotations", [])))
    if not airl_trajectories:
        raise FileNotFoundError("No compatible annotation files were found")
    return airl_trajectories, preference_trajectories, annotations_by_traj


def evaluate_policy_round(learner: PPO, venv: DummyVecEnv, episodes: int, max_steps: int) -> Dict[str, Any]:
    rewards = []
    success_count = 0
    for episode_idx in range(episodes):
        print(f"Evaluation episode {episode_idx + 1}/{episodes}")
        obs = venv.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        while not done and step_count < max_steps:
            action, _ = learner.predict(obs, deterministic=True)
            obs, reward, done_array, _ = venv.step(action)
            done = bool(done_array[0])
            episode_reward += float(reward[0])
            step_count += 1
        rewards.append(episode_reward)
        if episode_reward > 0:
            success_count += 1
        print(f"  Episode reward: {episode_reward:.3f}, Steps: {step_count}")
    return {
        "rewards": rewards,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "success_count": success_count,
    }


def validate_preference_stage_inputs(
    config: AIRLPrefBrushConfig,
    records: Sequence[Any],
    dataset_size: int,
) -> None:
    if len(records) < 2:
        raise ValueError("At least two valid fragments are required for preference refinement")
    if dataset_size < 1:
        raise ValueError("At least one preference pair is required for preference refinement")
    batch_size = min(max(1, dataset_size), config.reward_batch_size)
    if config.reward_minibatch_size is not None:
        if batch_size < config.reward_minibatch_size:
            raise ValueError("reward_batch_size must be at least reward_minibatch_size")
        if batch_size % config.reward_minibatch_size != 0:
            raise ValueError("reward_batch_size must be divisible by reward_minibatch_size")


def run_airl_stage(
    config: AIRLPrefBrushConfig,
    demo_trajectories: Sequence[types.Trajectory],
    sampled_files: Sequence[str],
    output_dir: str,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    prompt_for_stage_start("AIRL training", config.auto_reset)
    venv = DummyVecEnv([make_brush_env(config)])
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
        verbose=1,
        seed=config.seed,
        tensorboard_log=os.path.join(logs_dir, "policy"),
    )
    reward_net = build_reward_net(config)
    trainer = AIRL(
        demonstrations=list(demo_trajectories),
        demo_batch_size=config.demo_batch_size,
        gen_replay_buffer_capacity=config.gen_buff_size,
        n_disc_updates_per_round=config.n_disc,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        custom_logger=imit_logger.configure(os.path.join(logs_dir, "airl"), ["tensorboard", "stdout"]),
        allow_variable_horizon=True,
    )
    rewards_history: List[float] = []
    success_history: List[int] = []
    for round_idx in range(config.airl_training_rounds):
        print("\n" + "=" * 40)
        print(f"AIRL round {round_idx + 1}/{config.airl_training_rounds}")
        print("=" * 40)
        trainer.train(total_timesteps=config.airl_steps_per_round)
        evaluation = evaluate_policy_round(learner, venv, config.airl_eval_episodes, config.max_steps)
        rewards_history.append(evaluation["mean_reward"])
        success_history.append(evaluation["success_count"])
        if (round_idx + 1) % 5 == 0:
            learner.save(os.path.join(output_dir, f"checkpoint_round_{round_idx + 1}"))
    final_policy_path = os.path.join(output_dir, "airl_brush_final")
    learner.save(final_policy_path)
    reward_path = os.path.join(output_dir, "reward_net.pt")
    th.save(trainer.reward_train.state_dict(), reward_path)
    stats = {
        "sampled_files": [os.path.basename(path) for path in sampled_files],
        "trajectory_count": len(demo_trajectories),
        "reward_net_architecture": {
            "hidden_sizes": [config.reward_size, config.reward_size],
            "use_state": True,
            "use_action": True,
            "use_next_state": True,
        },
        "rewards_history": rewards_history,
        "success_history": success_history,
        "final_mean_reward": float(rewards_history[-1]) if rewards_history else 0.0,
        "best_reward": float(max(rewards_history)) if rewards_history else 0.0,
        "best_success_rate": max(success_history) / config.airl_eval_episodes if success_history else 0.0,
        "policy_path": final_policy_path,
        "reward_path": reward_path,
    }
    write_json(os.path.join(output_dir, "training_stats.json"), stats)
    venv.close()
    return stats


def run_preference_stage(
    config: AIRLPrefBrushConfig,
    preference_trajectories: Sequence[Any],
    annotations_by_traj: Sequence[Tuple[str, Sequence[Dict[str, Any]]]],
    sampled_files: Sequence[str],
    airl_reward_path: str,
    output_dir: str,
    rng: np.random.Generator,
) -> Tuple[DiscreteActionRewardNet, Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    preference_comparisons, _, imit_reward_logger, _ = load_preference_dependencies()
    records = build_fragment_records(preference_trajectories, annotations_by_traj, config)
    dataset, _, preferences = build_preference_dataset(records, config, rng)
    validate_preference_stage_inputs(config, records, len(dataset))
    reward_net = build_reward_net(config)
    reward_net.load_state_dict(th.load(airl_reward_path, map_location="cpu"))
    preference_model = preference_comparisons.PreferenceModel(
        model=reward_net,
        noise_prob=config.preference_noise_prob,
        discount_factor=config.preference_discount_factor,
        threshold=config.preference_model_threshold,
    )
    batch_size = min(max(1, len(dataset)), config.reward_batch_size)
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        preference_model=preference_model,
        loss=preference_comparisons.CrossEntropyRewardLoss(),
        rng=rng,
        batch_size=batch_size,
        minibatch_size=config.reward_minibatch_size,
        epochs=config.reward_epochs,
        lr=config.reward_learning_rate,
        custom_logger=imit_reward_logger.configure(os.path.join(logs_dir, "reward"), ["tensorboard", "stdout"]),
    )
    reward_trainer.train(dataset)
    reward_path = os.path.join(output_dir, "reward_net.pt")
    preference_path = os.path.join(output_dir, "preferences.pkl")
    th.save(reward_net.state_dict(), reward_path)
    dataset.save(preference_path)
    stats = {
        "sampled_files": [os.path.basename(path) for path in sampled_files],
        "trajectory_count": len(preference_trajectories),
        "fragment_count": len(records),
        "preference_count": int(len(dataset)),
        "prefer_first_count": int(np.sum(preferences == 1.0)),
        "prefer_second_count": int(np.sum(preferences == 0.0)),
        "tie_count": int(np.sum(preferences == 0.5)),
        "score_min": float(min(record.score for record in records)),
        "score_max": float(max(record.score for record in records)),
        "initialized_from_reward_path": airl_reward_path,
        "reward_path": reward_path,
        "preferences_path": preference_path,
    }
    write_json(os.path.join(output_dir, "training_stats.json"), stats)
    return reward_net, stats


def build_policy_learner(
    config: AIRLPrefBrushConfig,
    learned_reward_env: Any,
    output_dir: str,
    airl_policy_path: Optional[str],
):
    _, _, PPOClass, _ = load_policy_dependencies()
    if config.continue_policy_from_airl and airl_policy_path and os.path.exists(f"{airl_policy_path}.zip"):
        return PPOClass.load(airl_policy_path, env=learned_reward_env)
    return PPOClass(
        policy="MlpPolicy",
        env=learned_reward_env,
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
        verbose=1,
        seed=config.seed,
        tensorboard_log=os.path.join(output_dir, "logs", "policy"),
    )


def run_policy_stage(
    config: AIRLPrefBrushConfig,
    reward_net: DiscreteActionRewardNet,
    airl_policy_path: Optional[str],
    output_dir: str,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    prompt_for_stage_start("policy continuation", config.auto_reset)
    _, RewardVecEnvWrapper, _, DummyVecEnvClass = load_policy_dependencies()
    train_env = DummyVecEnvClass([make_brush_env(config)])
    learned_reward_env = RewardVecEnvWrapper(train_env, reward_net.predict_processed)
    eval_env = DummyVecEnvClass([make_brush_env(config)])
    learner = build_policy_learner(config, learned_reward_env, output_dir, airl_policy_path)
    rewards_history: List[float] = []
    success_history: List[int] = []
    for round_idx in range(config.policy_training_rounds):
        learner.learn(
            total_timesteps=config.policy_steps_per_round,
            reset_num_timesteps=round_idx == 0,
            tb_log_name="airlpref_brush",
            callback=learned_reward_env.make_log_callback(),
        )
        evaluation = evaluate_policy_round(learner, eval_env, config.policy_eval_episodes, config.max_steps)
        rewards_history.append(evaluation["mean_reward"])
        success_history.append(evaluation["success_count"])
        if (round_idx + 1) % 5 == 0:
            learner.save(os.path.join(output_dir, f"checkpoint_round_{round_idx + 1}"))
    final_policy_path = os.path.join(output_dir, "airlpref_policy_final")
    learner.save(final_policy_path)
    stats = {
        "initialized_from_airl_policy": bool(config.continue_policy_from_airl and airl_policy_path and os.path.exists(f"{airl_policy_path}.zip")),
        "rewards_history": rewards_history,
        "success_history": success_history,
        "final_mean_reward": float(rewards_history[-1]) if rewards_history else 0.0,
        "best_reward": float(max(rewards_history)) if rewards_history else 0.0,
        "best_success_rate": max(success_history) / config.policy_eval_episodes if success_history else 0.0,
        "policy_path": final_policy_path,
    }
    write_json(os.path.join(output_dir, "training_stats.json"), stats)
    eval_env.close()
    learned_reward_env.close()
    return stats


def train_airlpref_brush(config: AIRLPrefBrushConfig, demo_dir: str, output_dir: str) -> Dict[str, Any]:
    ensure_ros_node("train_airlpref_brush")
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(config.seed)
    th.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    sampled_files = get_annotation_files(demo_dir, config.demo_portion, rng)
    demo_trajectories, preference_trajectories, annotations_by_traj = load_selected_demonstrations(sampled_files)
    airl_stage_dir = os.path.join(output_dir, "airl_stage")
    pref_stage_dir = os.path.join(output_dir, "pref_stage")
    policy_stage_dir = os.path.join(output_dir, "policy_stage")
    airl_stats = run_airl_stage(config, demo_trajectories, sampled_files, airl_stage_dir)
    refined_reward_net, preference_stats = run_preference_stage(
        config,
        preference_trajectories,
        annotations_by_traj,
        sampled_files,
        airl_stats["reward_path"],
        pref_stage_dir,
        rng,
    )
    policy_stats: Dict[str, Any] = {
        "rewards_history": [],
        "success_history": [],
        "final_mean_reward": 0.0,
        "best_reward": 0.0,
        "best_success_rate": 0.0,
        "policy_path": None,
    }
    if config.run_policy_stage:
        policy_stats = run_policy_stage(config, refined_reward_net, airl_stats.get("policy_path"), policy_stage_dir)
    combined_stats = {
        "config": {key: value for key, value in config.__dict__.items() if not key.startswith("_")},
        "sampled_files": [os.path.basename(path) for path in sampled_files],
        "airl_stage": airl_stats,
        "preference_stage": preference_stats,
        "policy_stage": policy_stats,
    }
    write_json(os.path.join(output_dir, "training_stats.json"), combined_stats)
    return combined_stats


if __name__ == "__main__":
    ensure_ros_node("train_airlpref_brush")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    config = AIRLPrefBrushConfig(
        max_steps=64,
        airl_training_rounds=50,
        airl_steps_per_round=256,
        airl_eval_episodes=1,
        policy_training_rounds=20,
        policy_steps_per_round=256,
        policy_eval_episodes=1,
        auto_reset=False,
        demo_portion=0.5,
        run_policy_stage=True,
    )
    demo_dir = os.path.join(project_dir, "demonstrations")
    annotated_dir = os.path.join(demo_dir, "annotations")
    if not os.path.exists(annotated_dir):
        print(f"Error: Annotated demonstrations not found in {annotated_dir}")
        sys.exit(1)
    output_dir = os.path.join(project_dir, "experiments", config.experiment_name, "airlpref_trained")
    train_airlpref_brush(config, demo_dir, output_dir)
