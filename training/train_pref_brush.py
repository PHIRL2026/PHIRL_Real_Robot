import json
import os
import sys
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch as th

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PreferenceBrushConfig:
	max_steps: int = 64
	reward_size: int = 32
	demo_portion: float = 1.0
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
	learning_rate: float = 3e-4
	ppo_batch_size: int = 8
	n_steps: int = 32
	n_epochs: int = 10
	gamma: float = 0.97
	gae_lambda: float = 0.95
	ent_coef: float = 0.01
	max_grad_norm: float = 0.5
	training_rounds: int = 50
	steps_per_round: int = 256
	eval_episodes: int = 1
	auto_reset: bool = False
	train_policy: bool = True
	seed: int = 0


@dataclass(frozen=True)
class FragmentRecord:
	fragment: Any
	score: float
	source_file: str
	start_step: int
	end_step: int


def get_brush_spaces() -> Tuple[gym.spaces.Box, gym.spaces.Discrete]:
	observation_space = gym.spaces.Box(
		low=np.array([0.25, 0.1, 0.14], dtype=np.float32),
		high=np.array([0.55, 0.34, 0.45], dtype=np.float32),
		dtype=np.float32,
	)
	action_space = gym.spaces.Discrete(6)
	return observation_space, action_space


def load_preference_dependencies():
	from imitation.algorithms import preference_comparisons as preference_comparisons
	from imitation.data import types
	from imitation.util import logger as imit_logger
	from training.DiscreteActionRewardNet import DiscreteActionRewardNet

	return preference_comparisons, types, imit_logger, DiscreteActionRewardNet


def load_policy_dependencies():
	import rospy
	from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
	from stable_baselines3 import PPO
	from stable_baselines3.common.vec_env import DummyVecEnv

	return rospy, RewardVecEnvWrapper, PPO, DummyVecEnv


def make_brush_env(config: PreferenceBrushConfig):
	def _init():
		from env.BrushEnv import BrushEnv, StateSpace
		from env.BrushEnvWithManualReset import BrushEnvWithManualReset

		env = BrushEnv(state_space=StateSpace.CARTESIAN, max_steps=config.max_steps)
		env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_steps)
		env = BrushEnvWithManualReset(env, auto_reset=config.auto_reset)
		return env

	return _init


def get_annotation_files(demo_dir: str, demo_portion: float, rng: np.random.Generator) -> List[str]:
	annotated_dir = os.path.join(demo_dir, "annotations")
	if not os.path.exists(annotated_dir):
		raise FileNotFoundError(f"Annotated demonstrations not found: {annotated_dir}")
	demo_files = sorted(
		[
			os.path.join(annotated_dir, file_name)
			for file_name in os.listdir(annotated_dir)
			if file_name.endswith(".json")
			and "summary" not in file_name
			and (file_name.endswith("_progress.json") or file_name.startswith("demo_"))
		]
	)
	if not demo_files:
		raise FileNotFoundError(f"No annotation files found in {annotated_dir}")
	if demo_portion >= 1.0:
		return demo_files
	sample_size = max(1, int(np.ceil(len(demo_files) * demo_portion)))
	selected = rng.choice(np.array(demo_files), size=sample_size, replace=False)
	return sorted(selected.tolist())


def normalize_demo_arrays(demo: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	obs = np.asarray(demo.get("observations", []), dtype=np.float32)
	acts = np.asarray(demo.get("actions", []), dtype=np.int64)
	rews = np.asarray(demo.get("rewards", np.zeros(len(acts), dtype=np.float32)), dtype=np.float32)
	dones = np.asarray(demo.get("dones", np.zeros(len(acts), dtype=bool)), dtype=bool)
	if len(acts) == 0:
		raise ValueError("Demonstration must contain at least one action")
	target_obs_len = len(acts) + 1
	if len(obs) == 0:
		raise ValueError("Demonstration must contain at least one observation")
	if len(obs) < target_obs_len:
		pad = np.repeat(obs[-1:], target_obs_len - len(obs), axis=0)
		obs = np.concatenate([obs, pad], axis=0)
	elif len(obs) > target_obs_len:
		obs = obs[:target_obs_len]
	if len(rews) < len(acts):
		rews = np.pad(rews, (0, len(acts) - len(rews)), constant_values=0.0)
	else:
		rews = rews[: len(acts)]
	if len(dones) < len(acts):
		dones = np.pad(dones, (0, len(acts) - len(dones)), constant_values=False)
	else:
		dones = dones[: len(acts)]
	return obs, acts, rews.astype(np.float32), dones.astype(bool)


def load_annotated_trajectories(
	demo_dir: str,
	demo_portion: float,
	rng: np.random.Generator,
):
	preference_comparisons, types, _, _ = load_preference_dependencies()
	del preference_comparisons
	demo_files = get_annotation_files(demo_dir, demo_portion, rng)
	trajectories = []
	annotations_by_traj = []
	for file_path in demo_files:
		with open(file_path, "r") as file_handle:
			demo = json.load(file_handle)
		obs, acts, rews, dones = normalize_demo_arrays(demo)
		infos = np.array([{} for _ in range(len(acts))], dtype=object)
		trajectory = types.TrajectoryWithRew(
			obs=obs,
			acts=acts,
			infos=infos,
			terminal=bool(dones[-1]) if len(dones) else True,
			rews=rews,
		)
		trajectories.append(trajectory)
		annotations_by_traj.append((os.path.basename(file_path), demo.get("annotations", [])))
	return trajectories, annotations_by_traj


def build_fragment_records(
	trajectories: Sequence[Any],
	annotations_by_traj: Sequence[Tuple[str, Sequence[Dict[str, Any]]]],
	config: PreferenceBrushConfig,
) -> List[FragmentRecord]:
	_, types, _, _ = load_preference_dependencies()
	records: List[FragmentRecord] = []
	for trajectory, (source_file, annotations) in zip(trajectories, annotations_by_traj):
		for annotation in annotations:
			start_step = int(annotation.get("start_step", 0))
			end_step = int(annotation.get("end_step", 0))
			start_step = max(0, min(start_step, len(trajectory.acts) - 1))
			end_step = max(start_step + 1, min(end_step, len(trajectory.acts)))
			fragment_length = end_step - start_step
			if fragment_length < config.min_fragment_steps:
				continue
			start_progress = float(annotation.get("start_progress", 0.0))
			end_progress = float(annotation.get("end_progress", 0.0))
			if not np.isfinite(start_progress) or not np.isfinite(end_progress):
				continue
			progress_gain = end_progress - start_progress
			fragment_infos = None
			if trajectory.infos is not None:
				fragment_infos = trajectory.infos[start_step:end_step].copy()
			fragment = types.TrajectoryWithRew(
				obs=trajectory.obs[start_step : end_step + 1].copy(),
				acts=trajectory.acts[start_step:end_step].copy(),
				infos=fragment_infos,
				terminal=bool(trajectory.terminal and end_step == len(trajectory.acts)),
				rews=trajectory.rews[start_step:end_step].copy(),
			)
			records.append(
				FragmentRecord(
					fragment=fragment,
					score=float(progress_gain),
					source_file=source_file,
					start_step=start_step,
					end_step=end_step,
				)
			)
	if len(records) < 2:
		raise ValueError("At least two valid annotation fragments are required")
	return records


def build_preference_dataset(records: Sequence[FragmentRecord], config: PreferenceBrushConfig, rng: np.random.Generator):
	preference_comparisons, _, _, _ = load_preference_dependencies()
	pair_indices = list(combinations(range(len(records)), 2))
	if not pair_indices:
		raise ValueError("No fragment pairs available for preference learning")
	rng.shuffle(pair_indices)
	if config.max_preferences is not None:
		pair_indices = pair_indices[: config.max_preferences]
	fragment_pairs = []
	preferences = []
	for left_index, right_index in pair_indices:
		left_record = records[left_index]
		right_record = records[right_index]
		score_diff = left_record.score - right_record.score
		if score_diff > config.tie_threshold:
			label = 1.0
		elif score_diff < -config.tie_threshold:
			label = 0.0
		else:
			label = 0.5
		fragment_pairs.append((left_record.fragment, right_record.fragment))
		preferences.append(label)
	dataset = preference_comparisons.PreferenceDataset(max_size=config.comparison_queue_size)
	preference_array = np.asarray(preferences, dtype=np.float32)
	dataset.push(fragment_pairs, preference_array)
	return dataset, fragment_pairs, preference_array


def train_reward_model(
	config: PreferenceBrushConfig,
	demo_dir: str,
	output_dir: str,
	rng: np.random.Generator,
):
	preference_comparisons, _, imit_logger, DiscreteActionRewardNet = load_preference_dependencies()
	observation_space, action_space = get_brush_spaces()
	trajectories, annotations_by_traj = load_annotated_trajectories(demo_dir, config.demo_portion, rng)
	records = build_fragment_records(trajectories, annotations_by_traj, config)
	dataset, _, preferences = build_preference_dataset(records, config, rng)
	reward_net = DiscreteActionRewardNet(
		observation_space=observation_space,
		action_space=action_space,
		hidden_sizes=(config.reward_size, config.reward_size),
		use_state=True,
		use_action=True,
		use_next_state=True,
	)
	preference_model = preference_comparisons.PreferenceModel(
		model=reward_net,
		noise_prob=config.preference_noise_prob,
		discount_factor=config.preference_discount_factor,
		threshold=config.preference_model_threshold,
	)
	reward_logger = imit_logger.configure(
		os.path.join(output_dir, "logs", "reward"),
		["tensorboard", "stdout"],
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
		custom_logger=reward_logger,
	)
	reward_trainer.train(dataset)
	reward_net_path = os.path.join(output_dir, "reward_net.pt")
	preference_path = os.path.join(output_dir, "preferences.pkl")
	th.save(reward_net.state_dict(), reward_net_path)
	dataset.save(preference_path)
	reward_stats = {
		"trajectory_count": len(trajectories),
		"fragment_count": len(records),
		"preference_count": int(len(dataset)),
		"prefer_first_count": int(np.sum(preferences == 1.0)),
		"prefer_second_count": int(np.sum(preferences == 0.0)),
		"tie_count": int(np.sum(preferences == 0.5)),
		"score_min": float(min(record.score for record in records)),
		"score_max": float(max(record.score for record in records)),
		"reward_net_path": reward_net_path,
		"preferences_path": preference_path,
	}
	return reward_net, reward_stats


def ensure_ros_node():
	rospy, _, _, _ = load_policy_dependencies()
	if not rospy.core.is_initialized():
		rospy.init_node("train_pref_brush", anonymous=True)
	return rospy


def train_policy_with_learned_reward(
	config: PreferenceBrushConfig,
	reward_net: Any,
	output_dir: str,
):
	rospy, RewardVecEnvWrapper, PPO, DummyVecEnv = load_policy_dependencies()
	del rospy
	train_env = DummyVecEnv([make_brush_env(config)])
	learned_reward_env = RewardVecEnvWrapper(train_env, reward_net.predict_processed)
	eval_env = DummyVecEnv([make_brush_env(config)])
	learner = PPO(
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
	rewards_history = []
	success_history = []
	for round_idx in range(config.training_rounds):
		learner.learn(
			total_timesteps=config.steps_per_round,
			reset_num_timesteps=round_idx == 0,
			tb_log_name="pref_brush",
			callback=learned_reward_env.make_log_callback(),
		)
		eval_rewards = []
		eval_successes = 0
		for _ in range(config.eval_episodes):
			obs = eval_env.reset()
			done = False
			episode_reward = 0.0
			step_count = 0
			while not done and step_count < config.max_steps:
				action, _ = learner.predict(obs, deterministic=True)
				obs, reward, done_array, _ = eval_env.step(action)
				done = bool(done_array[0])
				episode_reward += float(reward[0])
				step_count += 1
			eval_rewards.append(episode_reward)
			if episode_reward > 0:
				eval_successes += 1
		mean_reward = float(np.mean(eval_rewards)) if eval_rewards else 0.0
		rewards_history.append(mean_reward)
		success_history.append(eval_successes)
		if (round_idx + 1) % 5 == 0:
			learner.save(os.path.join(output_dir, f"checkpoint_round_{round_idx + 1}"))
	final_path = os.path.join(output_dir, "pref_brush_final")
	learner.save(final_path)
	eval_env.close()
	learned_reward_env.close()
	return {
		"policy_path": final_path,
		"rewards_history": rewards_history,
		"success_history": success_history,
		"final_mean_reward": float(rewards_history[-1]) if rewards_history else 0.0,
		"best_reward": float(max(rewards_history)) if rewards_history else 0.0,
		"best_success_rate": max(success_history) / config.eval_episodes if success_history else 0.0,
	}


def train_pref_brush(config: PreferenceBrushConfig, demo_dir: str, output_dir: str):
	os.makedirs(output_dir, exist_ok=True)
	os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
	np.random.seed(config.seed)
	th.manual_seed(config.seed)
	rng = np.random.default_rng(config.seed)
	reward_net, reward_stats = train_reward_model(config, demo_dir, output_dir, rng)
	policy_stats: Dict[str, Any] = {
		"rewards_history": [],
		"success_history": [],
		"final_mean_reward": 0.0,
		"best_reward": 0.0,
		"best_success_rate": 0.0,
	}
	if config.train_policy:
		ensure_ros_node()
		policy_stats = train_policy_with_learned_reward(config, reward_net, output_dir)
	stats = {
		"config": {key: value for key, value in config.__dict__.items() if not key.startswith("_")},
		"reward_learning": reward_stats,
		"policy_training": policy_stats,
	}
	stats_path = os.path.join(output_dir, "training_stats.json")
	with open(stats_path, "w") as file_handle:
		json.dump(stats, file_handle, indent=2)
	return stats


if __name__ == "__main__":
	script_dir = os.path.dirname(os.path.abspath(__file__))
	project_dir = os.path.dirname(script_dir)
	config = PreferenceBrushConfig(
		max_steps=64,
		training_rounds=50,
		steps_per_round=256,
		eval_episodes=1,
		auto_reset=False,
		demo_portion=0.5,
		train_policy=True,
	)
	experiment_name = "EXP_1"
	demo_dir = os.path.join(project_dir, "demonstrations")
	annotated_dir = os.path.join(demo_dir, "annotations")
	if not os.path.exists(annotated_dir):
		print(f"Error: Annotated demonstrations not found in {annotated_dir}")
		sys.exit(1)
	output_dir = os.path.join(project_dir, "experiments", experiment_name, "pref_trained")
	train_pref_brush(config, demo_dir, output_dir)
