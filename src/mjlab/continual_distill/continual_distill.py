#!/usr/bin/env python3
"""
Continual policy distillation with Synaptic Intelligence (SI) for mjlab environments.

This script extends single-task distillation to sequential tasks defined via YAML.
Each task provides an offline dataset of teacher targets; the student shares a
multi-head MLP and accumulates SI penalties to mitigate forgetting.

Key differences from Brax version:
- Uses mjlab environments instead of Brax
- Loads PyTorch teacher checkpoints and converts to JAX
- Evaluates using mjlab's episode_success metric
- Each task has its own observation normalizer
- No video logging (for now)
"""

import argparse
import functools
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

print("-------------- Setting up environment variables -----------")
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import yaml
from flax import struct
from flax.training import train_state
from tqdm import trange

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.continual_distill.utils import (
    ObservationNormalizer,
    StudentPolicy,
    TeacherPolicy,
)

try:
    import wandb
except ImportError:
    wandb = None


class StudentTrainStateSI(train_state.TrainState):
    """Training state with Synaptic Intelligence tracking."""
    normalizer_params: Any
    prev_step_params_flat: jnp.ndarray
    snapshot_params_flat: jnp.ndarray
    omega: jnp.ndarray
    omega_total: jnp.ndarray
    ravel_fn: Any = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    num_tasks: int = struct.field(pytree_node=False)
    student_min_std: float = struct.field(pytree_node=False)


def _slugify(name: str) -> str:
    """Convert task name to slug."""
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_") or "task"


def gaussian_kl(
    teacher_mean: jnp.ndarray,
    teacher_logstd: jnp.ndarray,
    student_mean: jnp.ndarray,
    student_logstd: jnp.ndarray,
) -> jnp.ndarray:
    """KL(teacher || student) for diagonal Gaussians; returns per-sample values."""
    var_teacher = jnp.exp(2.0 * teacher_logstd)
    var_student = jnp.exp(2.0 * student_logstd)
    log_std_ratio = student_logstd - teacher_logstd
    squared_diff = jnp.square(student_mean - teacher_mean)
    kl = log_std_ratio + (var_teacher + squared_diff) / (2.0 * var_student) - 0.5
    return jnp.sum(kl, axis=-1)


def init_wandb(args, config: Dict[str, Any]):
    """Initialize Weights & Biases tracking."""
    if not args.track:
        return None
    if wandb is None:
        print("[continual_distill] wandb not installed; disabling tracking.")
        return None
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.run_name,
        config=config,
    )
    return run


def compute_student_distribution(
    apply_fn,
    params,
    normalizer_params,
    obs,
    student_min_std: float,
    action_dim: int,
    task_idx: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get student distribution parameters for specific task.

    Args:
        apply_fn: Network apply function
        params: Network parameters
        normalizer_params: Observation normalizer dict with 'mean' and 'std'
        obs: Observations
        student_min_std: Minimum std for student policy
        action_dim: Action dimension
        task_idx: Task index

    Returns:
        Tuple of (mean, logstd)
    """
    # Normalize observations
    if isinstance(normalizer_params, dict):
        normalized_obs = (obs - normalizer_params['mean']) / (normalizer_params['std'] + 1e-8)
    else:
        normalized_obs = normalizer_params.normalize(obs)

    # Forward pass through network
    logits = apply_fn(params, normalized_obs)

    # Extract head for this task
    head_dim = 2 * action_dim
    start = task_idx * head_dim
    head_logits = jax.lax.dynamic_slice_in_dim(logits, start_index=start, slice_size=head_dim, axis=-1)

    # Split into mean and scale parameters
    loc, scale_params = jnp.split(head_logits, 2, axis=-1)
    std = jax.nn.softplus(scale_params) + student_min_std
    log_std = jnp.log(std)

    return loc, log_std


@jax.jit
def train_step_si(
    state: StudentTrainStateSI,
    batch_obs: jnp.ndarray,
    batch_teacher_mean: jnp.ndarray,
    batch_teacher_logstd: jnp.ndarray,
    task_idx: jnp.ndarray,
    si_coeff: float,
) -> Tuple[StudentTrainStateSI, jnp.ndarray]:
    """Single training step with SI regularization.

    Args:
        state: Training state with SI buffers
        batch_obs: Batch of observations
        batch_teacher_mean: Teacher means
        batch_teacher_logstd: Teacher logstds
        task_idx: Current task index
        si_coeff: SI regularization coefficient

    Returns:
        Tuple of (updated state, metrics)
    """
    task_idx = jnp.asarray(task_idx, dtype=jnp.int32)
    si_scale = jnp.where(
        task_idx > 0,
        jnp.asarray(si_coeff, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )

    def loss_fn(params):
        student_mean, student_logstd = compute_student_distribution(
            state.apply_fn,
            params,
            state.normalizer_params,
            batch_obs,
            state.student_min_std,
            state.action_dim,
            task_idx,
        )
        kl_vals = gaussian_kl(
            batch_teacher_mean, batch_teacher_logstd, student_mean, student_logstd
        )
        dist_loss = jnp.mean(kl_vals)
        params_flat = state.ravel_fn(params)
        surrogate = jnp.sum(
            state.omega_total * jnp.square(params_flat - state.snapshot_params_flat)
        )
        total_loss = dist_loss + si_scale * surrogate
        return total_loss, (dist_loss, surrogate)

    (total_loss, (dist_loss, surrogate)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    params_flat_after = state.ravel_fn(new_state.params)
    grads_flat = state.ravel_fn(grads)
    omega_update = state.omega - grads_flat * (params_flat_after - state.prev_step_params_flat)
    new_state = new_state.replace(
        prev_step_params_flat=params_flat_after,
        omega=omega_update,
    )
    metrics = jnp.stack(
        [total_loss, dist_loss, si_scale * surrogate, surrogate],
        axis=0,
    )
    return new_state, metrics


@functools.partial(jax.jit, static_argnums=(5,))
def train_epoch_si(
    state: StudentTrainStateSI,
    rng: jax.random.PRNGKey,
    obs: jnp.ndarray,
    teacher_mean: jnp.ndarray,
    teacher_logstd: jnp.ndarray,
    batch_size: int,
    si_coeff: float,
    task_idx: jnp.ndarray,
) -> Tuple[StudentTrainStateSI, jax.random.PRNGKey, jnp.ndarray, jnp.ndarray]:
    """Train for one epoch with SI regularization.

    Args:
        state: Training state
        rng: Random key
        obs: All observations
        teacher_mean: All teacher means
        teacher_logstd: All teacher logstds
        batch_size: Mini-batch size (static)
        si_coeff: SI coefficient
        task_idx: Current task index

    Returns:
        Tuple of (updated state, updated rng, mean metrics, last batch metrics)
    """
    num_samples = obs.shape[0]
    num_batches = num_samples // batch_size
    if num_batches == 0:
        raise ValueError("Training set must have at least one full batch.")

    rng, perm_key = jax.random.split(rng)
    permutation = jax.random.permutation(perm_key, num_samples)
    shuffled_obs = jnp.take(obs, permutation, axis=0)
    shuffled_mean = jnp.take(teacher_mean, permutation, axis=0)
    shuffled_logstd = jnp.take(teacher_logstd, permutation, axis=0)

    batch_elems = num_batches * batch_size
    shuffled_obs = shuffled_obs[:batch_elems]
    shuffled_mean = shuffled_mean[:batch_elems]
    shuffled_logstd = shuffled_logstd[:batch_elems]

    obs_batches = shuffled_obs.reshape((num_batches, batch_size, shuffled_obs.shape[-1]))
    mean_batches = shuffled_mean.reshape((num_batches, batch_size, shuffled_mean.shape[-1]))
    logstd_batches = shuffled_logstd.reshape((num_batches, batch_size, shuffled_logstd.shape[-1]))

    def batch_update(carry, batch):
        train_state, loss_sums = carry
        batch_obs, batch_mean, batch_logstd = batch
        train_state, metrics = train_step_si(
            train_state,
            batch_obs,
            batch_mean,
            batch_logstd,
            task_idx,
            si_coeff,
        )
        loss_sums = loss_sums + metrics
        return (train_state, loss_sums), metrics

    init_carry = (state, jnp.zeros(4, dtype=jnp.float32))
    (final_state, total_metrics), metrics_per_batch = jax.lax.scan(
        batch_update,
        init_carry,
        (obs_batches, mean_batches, logstd_batches),
    )

    mean_metrics = total_metrics / num_batches
    last_metrics = metrics_per_batch[-1]
    return final_state, rng, mean_metrics, last_metrics


@jax.jit
def dataset_kl_loss_si(
    state: StudentTrainStateSI,
    obs: jnp.ndarray,
    teacher_mean: jnp.ndarray,
    teacher_logstd: jnp.ndarray,
    task_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Compute KL loss on entire dataset."""
    mean_pred, logstd_pred = compute_student_distribution(
        state.apply_fn,
        state.params,
        state.normalizer_params,
        obs,
        state.student_min_std,
        state.action_dim,
        task_idx,
    )
    kl_values = gaussian_kl(teacher_mean, teacher_logstd, mean_pred, logstd_pred)
    return jnp.mean(kl_values)


def evaluate_environment(
    task: Dict[str, Any],
    state: StudentTrainStateSI,
    task_idx: int,
    task_name: str,
    num_episodes: int,
    episode_length: int,
    seed: int,
    wandb_run=None,
    log_to_wandb: bool = False,
    global_step: Optional[int] = None,
    log_epoch: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate student policy in mjlab environment.

    Args:
        task: Task dictionary containing env and teacher info
        state: Student training state
        task_idx: Task index
        task_name: Task name for logging
        num_episodes: Number of episodes to run
        episode_length: Maximum episode length
        seed: Random seed
        wandb_run: Wandb run object
        log_to_wandb: Whether to log to wandb
        global_step: Global training step
        log_epoch: Current epoch

    Returns:
        Dictionary of evaluation metrics
    """
    if num_episodes <= 0:
        return {
            "env_loss": 0.0,
            "teacher_return": 0.0,
            "student_return": 0.0,
            "teacher_success": 0.0,
            "student_success": 0.0,
        }

    env = task["env"]
    teacher_policy = task["teacher_policy"]
    teacher_params = task["teacher_params"]
    teacher_std = task["teacher_std"]
    teacher_normalizer = task["teacher_normalizer"]

    task_idx_arr = jnp.asarray(task_idx, dtype=jnp.int32)

    # JIT compile action functions
    @jax.jit
    def get_teacher_actions_batch(obs_jax):
        """Get teacher actions for batch of observations."""
        normalized_obs = teacher_normalizer.normalize(obs_jax)
        action_mean = teacher_policy.actor.apply(teacher_params, normalized_obs)
        return action_mean

    @jax.jit
    def get_student_actions_batch(obs_jax):
        """Get student actions for batch of observations."""
        mean, _ = compute_student_distribution(
            state.apply_fn,
            state.params,
            state.normalizer_params,
            obs_jax,
            state.student_min_std,
            state.action_dim,
            task_idx_arr,
        )
        return mean

    # Teacher rollouts - parallel episodes (run full episode length, no early termination)
    obs, _ = env.reset()
    num_envs = env.num_envs
    episode_returns = np.zeros(num_envs)
    teacher_total_kl = 0.0
    teacher_total_steps = 0

    # Run for full episode length (no early termination)
    for step in range(episode_length):
        # Get observation and convert to JAX
        obs_array = obs['policy'].cpu().numpy()
        obs_jax = jnp.array(obs_array)

        # Get teacher and student actions
        teacher_action_jax = get_teacher_actions_batch(obs_jax)
        student_action_jax = get_student_actions_batch(obs_jax)

        # Compute KL (using teacher std)
        teacher_std_jax = jnp.array(teacher_std)
        teacher_logstd = jnp.log(teacher_std_jax)
        _, student_logstd = compute_student_distribution(
            state.apply_fn, state.params, state.normalizer_params,
            obs_jax, state.student_min_std, state.action_dim, task_idx_arr,
        )
        kl_vals = gaussian_kl(teacher_action_jax, teacher_logstd, student_action_jax, student_logstd)
        teacher_total_kl += float(jnp.sum(kl_vals))
        teacher_total_steps += obs_jax.shape[0]

        # Step environment
        teacher_action_torch = torch.from_numpy(np.array(teacher_action_jax))
        obs, reward, terminated, truncated, info = env.step(teacher_action_torch)
        episode_returns += reward.cpu().numpy()

    # Extract final success values from info["log"] (populated when episode ends)
    teacher_successes = np.zeros(num_envs)
    if "log" in info and isinstance(info["log"], dict):
        for key, value in info["log"].items():
            if "episode_success" in key.lower():
                teacher_successes = value.cpu().numpy() if hasattr(value, 'cpu') else np.array(value)
                break

    teacher_episode_returns = episode_returns.tolist()
    teacher_successes = teacher_successes.tolist()

    # Student rollouts - parallel episodes (run full episode length, no early termination)
    obs, _ = env.reset()
    episode_returns = np.zeros(num_envs)

    # Run for full episode length (no early termination)
    for step in range(episode_length):
        obs_array = obs['policy'].cpu().numpy()
        obs_jax = jnp.array(obs_array)

        student_action_jax = get_student_actions_batch(obs_jax)
        student_action_torch = torch.from_numpy(np.array(student_action_jax))
        obs, reward, terminated, truncated, info = env.step(student_action_torch)
        episode_returns += reward.cpu().numpy()

    # Extract final success values from info["log"] (populated when episode ends)
    student_successes = np.zeros(num_envs)
    if "log" in info and isinstance(info["log"], dict):
        for key, value in info["log"].items():
            if "episode_success" in key.lower():
                student_successes = value.cpu().numpy() if hasattr(value, 'cpu') else np.array(value)
                break

    student_episode_returns = episode_returns.tolist()
    student_successes = student_successes.tolist()

    # Compute metrics
    avg_env_kl = teacher_total_kl / max(teacher_total_steps, 1)
    metrics = {
        "env_loss": float(avg_env_kl),
        "teacher_return": float(np.mean(teacher_episode_returns)),
        "student_return": float(np.mean(student_episode_returns)),
        "teacher_success": float(np.mean(teacher_successes)),
        "student_success": float(np.mean(student_successes)),
    }

    if log_to_wandb and wandb_run is not None:
        payload = {
            f"EnvEval/task_{task_idx}_{task_name}/ts_loss/total": metrics["env_loss"],
            f"Rewards/task_{task_idx}_{task_name}": metrics["student_return"],
            f"Accuracy/task_{task_idx}_{task_name}": metrics["student_success"],
        }
        if log_epoch is not None:
            payload["Training/epoch"] = log_epoch
        wandb_run.log(payload, step=global_step)

    return metrics


def evaluate_all_tasks_offline(
    state: StudentTrainStateSI,
    task_buffers: List[Dict[str, Any]],
    current_task_idx: int,
    wandb_run,
    global_step: int,
    epoch: Optional[int] = None,
) -> None:
    """Evaluate all tasks from 0 to current_task_idx on offline data."""
    for eval_task_idx in range(len(task_buffers)):
        task_data = task_buffers[eval_task_idx]
        eval_task_name = task_data.get("task_name", f"task_{eval_task_idx}")

        if eval_task_idx > current_task_idx:
            # Log placeholder for future tasks
            if wandb_run is not None:
                payload = {
                    f"TotalData_Loss/task_{eval_task_idx}_{eval_task_name}/total": -0.01,
                    f"TestData_Loss/task_{eval_task_idx}_{eval_task_name}/total": -0.01,
                }
                if epoch is not None:
                    payload["Training/epoch"] = epoch
                wandb_run.log(payload, step=global_step)
            continue

        eval_task_idx_arr = jnp.asarray(eval_task_idx, dtype=jnp.int32)

        # Evaluate on train split
        train_obs = task_data["train_obs"]
        train_mean = task_data["train_mean"]
        train_logstd = task_data["train_logstd"]
        train_loss = float(
            dataset_kl_loss_si(
                state,
                train_obs,
                train_mean,
                train_logstd,
                eval_task_idx_arr,
            )
        )

        # Evaluate on test split
        test_obs = task_data["test_obs"]
        test_mean = task_data["test_mean"]
        test_logstd = task_data["test_logstd"]
        test_size = task_data["test_size"]

        payload = {f"TotalData_Loss/task_{eval_task_idx}_{eval_task_name}/total": train_loss}
        if test_size > 0:
            test_loss = float(
                dataset_kl_loss_si(
                    state,
                    test_obs,
                    test_mean,
                    test_logstd,
                    eval_task_idx_arr,
                )
            )
            payload[f"TestData_Loss/task_{eval_task_idx}_{eval_task_name}/total"] = test_loss

        if epoch is not None:
            payload["Training/epoch"] = epoch

        if wandb_run is not None:
            wandb_run.log(payload, step=global_step)


def evaluate_all_tasks_env(
    state: StudentTrainStateSI,
    task_buffers: List[Dict[str, Any]],
    current_task_idx: int,
    num_episodes: int,
    episode_length: int,
    seed: int,
    wandb_run,
    global_step: int,
    epoch: Optional[int] = None,
) -> None:
    """Evaluate all tasks from 0 to current_task_idx in the environment."""
    if num_episodes <= 0:
        return

    for eval_task_idx in range(len(task_buffers)):
        task_data = task_buffers[eval_task_idx]
        eval_task_name = task_data.get("task_name", f"task_{eval_task_idx}")

        if eval_task_idx > current_task_idx:
            # Log placeholder for future tasks
            if wandb_run is not None:
                payload = {
                    f"EnvEval/task_{eval_task_idx}_{eval_task_name}/ts_loss/total": -0.01,
                    f"Rewards/task_{eval_task_idx}_{eval_task_name}": -0.001,
                    f"Accuracy/task_{eval_task_idx}_{eval_task_name}": -0.001,
                }
                if epoch is not None:
                    payload["Training/epoch"] = epoch
                wandb_run.log(payload, step=global_step)
            continue

        env_metrics = evaluate_environment(
            task=task_data,
            state=state,
            task_idx=eval_task_idx,
            task_name=eval_task_name,
            num_episodes=num_episodes,
            episode_length=episode_length,
            seed=seed + eval_task_idx * 1000,
            wandb_run=wandb_run,
            log_to_wandb=wandb_run is not None,
            global_step=global_step,
            log_epoch=epoch,
        )


def consolidate_si_state(state: StudentTrainStateSI, epsilon: float) -> StudentTrainStateSI:
    """Consolidate SI buffers after task completion."""
    params_flat = state.ravel_fn(state.params)
    delta = params_flat - state.snapshot_params_flat
    denom = jnp.square(delta) + jnp.asarray(epsilon, dtype=jnp.float32)
    omega_task = state.omega / denom
    omega_task = jnp.where(jnp.isfinite(omega_task), omega_task, 0.0)
    omega_total = jnp.maximum(0.0, state.omega_total + omega_task)
    zeros = jnp.zeros_like(state.omega)
    return state.replace(
        omega=zeros,
        omega_total=omega_total,
        snapshot_params_flat=params_flat,
        prev_step_params_flat=params_flat,
    )


def _load_yaml_config(config_path: Path) -> List[Dict[str, Any]]:
    """Load task configuration from YAML file."""
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    tasks = config.get("task_info", [])
    if not tasks:
        raise ValueError(f"No task_info entries found in {config_path}")
    return tasks


def _load_dataset_and_teacher(dataset_folder: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load dataset and teacher from folder containing data.pkl and teacher.pkl.

    Args:
        dataset_folder: Path to folder containing data.pkl and teacher.pkl

    Returns:
        dataset: Dict with observations, action_targets, metadata
        teacher_info: Dict with JAX params, normalizer, architecture
    """
    if not dataset_folder.exists():
        raise FileNotFoundError(f"Dataset folder {dataset_folder} does not exist")

    data_path = dataset_folder / "data.pkl"
    teacher_path = dataset_folder / "teacher.pkl"

    if not data_path.exists():
        raise FileNotFoundError(f"data.pkl not found in {dataset_folder}")
    if not teacher_path.exists():
        raise FileNotFoundError(f"teacher.pkl not found in {dataset_folder}")

    # Load dataset
    with data_path.open("rb") as f:
        data = pickle.load(f)
    observations = np.asarray(data["observations"], dtype=np.float32)
    action_targets = np.asarray(data["action_targets"], dtype=np.float32)
    dataset = {
        "observations": observations,
        "action_targets": action_targets,
        "metadata": data.get("metadata", {}),
    }

    # Load teacher (already in JAX format!)
    with teacher_path.open("rb") as f:
        teacher_info = pickle.load(f)

    # Convert numpy back to JAX arrays
    def numpy_to_jax(tree):
        """Convert numpy arrays in pytree to JAX arrays."""
        import jax.tree_util as tree_util
        return tree_util.tree_map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, tree)

    teacher_info['jax_params'] = numpy_to_jax(teacher_info['jax_params'])
    teacher_info['action_std'] = jnp.array(teacher_info['action_std'])
    teacher_info['obs_normalizer_mean'] = jnp.array(teacher_info['obs_normalizer_mean'])
    teacher_info['obs_normalizer_std'] = jnp.array(teacher_info['obs_normalizer_std'])

    return dataset, teacher_info


def _build_task_normalizer(observations: np.ndarray) -> Dict[str, jnp.ndarray]:
    """Build observation normalizer parameters for single task.

    Returns a dictionary with 'mean' and 'std' JAX arrays (JIT-compatible).
    """
    if observations.size == 0:
        obs_dim = 1
        mean = np.zeros(obs_dim, dtype=np.float32)
        std = np.ones(obs_dim, dtype=np.float32)
    else:
        mean = observations.mean(axis=0).astype(np.float32)
        std = observations.std(axis=0).astype(np.float32)
        std = np.maximum(std, 1e-6)

    return {
        'mean': jnp.array(mean),
        'std': jnp.array(std),
    }


def _split_dataset_for_task(
    observations: np.ndarray,
    teacher_mean: np.ndarray,
    teacher_logstd: np.ndarray,
    train_fraction: float,
    episode_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into train and test."""
    num_samples = observations.shape[0]
    if num_samples == 0:
        raise ValueError("Dataset is empty; cannot train.")
    train_fraction = float(np.clip(train_fraction, 0.0, 1.0))
    train_samples = int(train_fraction * num_samples)
    if episode_length > 0:
        train_samples = (train_samples // episode_length) * episode_length
    train_samples = max(min(train_samples, num_samples), 1)
    train_obs = observations[:train_samples]
    train_mean = teacher_mean[:train_samples]
    train_logstd = teacher_logstd[:train_samples]
    test_obs = observations[train_samples:]
    test_mean = teacher_mean[train_samples:]
    test_logstd = teacher_logstd[train_samples:]
    return train_obs, train_mean, train_logstd, test_obs, test_mean, test_logstd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Continual policy distillation with Synaptic Intelligence (SI) for mjlab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML file describing the sequence of tasks.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size for SGD/Adam updates.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of each dataset used for training.")
    parser.add_argument("--student-min-std", type=float, default=1e-3, help="Minimum std for student policy outputs.")
    parser.add_argument("--si-coeff", type=float, default=1.0, help="Regularization coefficient for SI surrogate.")
    parser.add_argument("--si-epsilon", type=float, default=1e-3, help="Stability term for SI consolidation.")
    parser.add_argument("--eval-every", type=int, default=50, help="Frequency (in epochs) of offline evaluation.")
    parser.add_argument("--env-eval-every", type=int, default=50, help="Frequency (in epochs) of environment evaluation; 0 => match --eval-every.")
    parser.add_argument("--env-eval-episodes", type=int, default=4, help="Number of episodes for environment evaluation.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--track", action="store_true", default=True, help="Enable Weights & Biases logging.")
    parser.add_argument("--no-track", action="store_false", dest="track", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="continual_rl_mjlab", help="wandb project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity/team.")
    parser.add_argument("--wandb-mode", type=str, default="online", help="wandb mode (online/offline/disabled).")
    parser.add_argument("--run-name", type=str, default=None, help="Optional wandb run name.")
    parser.add_argument("--device", type=str, default=None, help="Device for mjlab environments (default: cuda:0 if available).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_eval_every = args.env_eval_every if args.env_eval_every > 0 else max(args.eval_every, 1)

    task_configs = _load_yaml_config(args.config)

    # Load datasets and teachers
    datasets: List[Dict[str, Any]] = []
    teacher_infos: List[Dict[str, Any]] = []
    for task in task_configs:
        dataset_folder = Path(task["dataset_folder"]).expanduser()
        data, teacher_info = _load_dataset_and_teacher(dataset_folder)
        datasets.append(data)
        teacher_infos.append(teacher_info)

    # Get dimensions from first task
    obs_dim_values = {int(task["obs_dim"]) for task in task_configs}
    if len(obs_dim_values) != 1:
        raise ValueError(f"All tasks must share the same observation dimension; got {obs_dim_values}")
    action_dim_values = {int(task["action_dim"]) for task in task_configs}
    if len(action_dim_values) != 1:
        raise ValueError(f"All tasks must share the same action dimension; got {action_dim_values}")

    obs_dim = obs_dim_values.pop()
    action_dim = action_dim_values.pop()
    num_tasks = len(task_configs)

    # Create student policy
    student = StudentPolicy(
        obs_size=obs_dim,
        action_size=action_dim,
        num_tasks=num_tasks,
        hidden_dims=(512, 256, 128),  # Match teacher architecture
    )
    init_key = jax.random.PRNGKey(args.seed)
    params = student.init(init_key)
    flat_params = student.flatten_tree(params)
    optimizer = optax.adam(args.learning_rate)

    # Create initial normalizer (will be replaced per-task)
    init_normalizer = _build_task_normalizer(datasets[0]["observations"])

    state = StudentTrainStateSI.create(
        apply_fn=student.network.apply,
        params=params,
        tx=optimizer,
        normalizer_params=init_normalizer,
        prev_step_params_flat=flat_params,
        snapshot_params_flat=jnp.zeros_like(flat_params),
        omega=jnp.zeros_like(flat_params),
        omega_total=jnp.zeros_like(flat_params),
        ravel_fn=student.flatten_tree,
        action_dim=action_dim,
        num_tasks=num_tasks,
        student_min_std=args.student_min_std,
    )

    # Prepare task buffers
    task_summaries: List[Dict[str, Any]] = []
    task_buffers: List[Dict[str, Any]] = []

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    for task_idx, task_cfg in enumerate(task_configs):
        dataset = datasets[task_idx]
        observations = dataset["observations"]
        action_targets = dataset["action_targets"]
        teacher_mean = action_targets[:, :action_dim]
        teacher_logstd = action_targets[:, action_dim:]

        # Build task-specific normalizer
        task_normalizer = _build_task_normalizer(observations)

        train_obs_np, train_mean_np, train_logstd_np, test_obs_np, test_mean_np, test_logstd_np = _split_dataset_for_task(
            observations,
            teacher_mean,
            teacher_logstd,
            args.train_fraction,
            int(task_cfg.get("ep_len", 0)),
        )

        train_size = int(train_obs_np.shape[0])
        test_size = int(test_obs_np.shape[0])
        if train_size < args.batch_size:
            raise ValueError(
                f"Task {task_idx} training set ({train_size}) smaller than batch size ({args.batch_size})."
            )
        train_steps_per_epoch = max(1, train_size // args.batch_size)

        task_name = task_cfg.get("task_name", f"task_{task_idx}")
        task_slug = _slugify(task_name) or f"task_{task_idx}"
        dataset_folder_str = str(Path(task_cfg["dataset_folder"]).expanduser())
        num_epochs = int(task_cfg.get("num_epochs", 500))  # Default to 500 if not specified in YAML
        episode_length = int(task_cfg.get("ep_len", 100))

        # Load teacher from preloaded teacher_info (already in JAX format!)
        teacher_info = teacher_infos[task_idx]
        print(f"\n[Task {task_idx}] Loading teacher from: {dataset_folder_str}/teacher.pkl")

        # Create teacher policy and normalizer from loaded info
        teacher_policy = TeacherPolicy(
            obs_size=obs_dim,
            action_size=action_dim,
            hidden_dims=tuple(teacher_info['hidden_dims']),
        )
        teacher_params = teacher_info['jax_params']
        teacher_std = teacher_info['action_std']
        teacher_normalizer = ObservationNormalizer(
            mean=teacher_info['obs_normalizer_mean'],
            std=teacher_info['obs_normalizer_std'],
        )

        # Create mjlab environment for evaluation
        env_id = task_cfg["env_id"]
        env_cfg = load_env_cfg(env_id, test=True)
        env_cfg.scene.num_envs = args.env_eval_episodes
        env_cfg.seed = args.seed
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

        # Use environment's actual episode length for evaluation
        env_episode_length = env.max_episode_length
        print(f"[INFO] Task {task_idx}: Using environment episode length: {env_episode_length} steps")
        episode_length = env_episode_length

        task_buffers.append(
            {
                "train_obs": jnp.asarray(train_obs_np),
                "train_mean": jnp.asarray(train_mean_np),
                "train_logstd": jnp.asarray(train_logstd_np),
                "test_obs": jnp.asarray(test_obs_np),
                "test_mean": jnp.asarray(test_mean_np),
                "test_logstd": jnp.asarray(test_logstd_np),
                "task_name": task_name,
                "task_slug": task_slug,
                "train_size": train_size,
                "test_size": test_size,
                "train_steps_per_epoch": train_steps_per_epoch,
                "num_epochs": num_epochs,
                "dataset_folder": dataset_folder_str,
                "task_normalizer": task_normalizer,
                "env": env,
                "episode_length": episode_length,
                "teacher_policy": teacher_policy,
                "teacher_params": teacher_params,
                "teacher_std": teacher_std,
                "teacher_normalizer": teacher_normalizer,
            }
        )

        task_summaries.append(
            {
                "index": task_idx,
                "name": task_name,
                "slug": task_slug,
                "env_id": env_id,
                "dataset_folder": dataset_folder_str,
                "train_size": train_size,
                "test_size": test_size,
                "train_steps_per_epoch": train_steps_per_epoch,
                "num_epochs": num_epochs,
                "episode_length": episode_length,
            }
        )

    print("=" * 80)
    print("Continual Policy Distillation with Synaptic Intelligence (SI) - mjlab")
    print("=" * 80)
    print(f"Tasks: {num_tasks} | Obs dim: {obs_dim} | Action dim: {action_dim}")
    print(f"Learning rate: {args.learning_rate} | Batch size: {args.batch_size}")
    print(f"SI coef: {args.si_coeff} | SI epsilon: {args.si_epsilon}")

    args.run_name = args.run_name or f"continual_distill_{int(time.time())}"

    total_updates_planned = sum(tb["num_epochs"] * tb["train_steps_per_epoch"] for tb in task_buffers)
    run_config = {
        "num_tasks": num_tasks,
        "config_path": str(args.config),
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "student_min_std": args.student_min_std,
        "train_fraction": args.train_fraction,
        "si_coeff": args.si_coeff,
        "si_epsilon": args.si_epsilon,
        "eval_every": args.eval_every,
        "env_eval_every": env_eval_every,
        "env_eval_episodes": args.env_eval_episodes,
        "tasks": task_summaries,
    }
    wandb_run = init_wandb(args, run_config)

    start_time = time.time()
    train_rng = jax.random.PRNGKey(args.seed + 1)
    global_step = 0
    offline_eval_every = max(args.eval_every, 1)

    for task_idx, task_data in enumerate(task_buffers):
        task_name = task_data["task_name"]
        print("\n" + "-" * 80)
        print(f"[Task {task_idx}] {task_name}")
        print(f"Dataset folder: {task_data['dataset_folder']}")

        # Update normalizer for this task
        state = state.replace(normalizer_params=task_data["task_normalizer"])

        train_obs = task_data["train_obs"]
        train_mean = task_data["train_mean"]
        train_logstd = task_data["train_logstd"]
        test_obs = task_data["test_obs"]
        test_mean = task_data["test_mean"]
        test_logstd = task_data["test_logstd"]
        train_size = task_data["train_size"]
        test_size = task_data["test_size"]
        train_steps_per_epoch = task_data["train_steps_per_epoch"]
        print(f"Train samples: {train_size} | Test samples: {test_size}")

        # Reset optimizer for new task
        optimizer = optax.adam(args.learning_rate)
        state = state.replace(
            tx=optimizer,
            opt_state=optimizer.init(state.params),
            prev_step_params_flat=student.flatten_tree(state.params),
        )

        task_num_epochs = task_data["num_epochs"]

        # Initial evaluation at step 0 for this task (before training)
        print(f"\n[Task {task_idx}] Initial Evaluation (Step 0 - Before Training)")

        # Evaluate all tasks on offline data at step 0
        evaluate_all_tasks_offline(
            state=state,
            task_buffers=task_buffers,
            current_task_idx=task_idx,
            wandb_run=wandb_run,
            global_step=global_step,
            epoch=0,
        )

        # Evaluate all tasks in environment at step 0
        if args.env_eval_episodes > 0:
            evaluate_all_tasks_env(
                state=state,
                task_buffers=task_buffers,
                current_task_idx=task_idx,
                num_episodes=args.env_eval_episodes,
                episode_length=task_data["episode_length"],
                seed=args.seed,
                wandb_run=wandb_run,
                global_step=global_step,
                epoch=0,
            )
            print(f"      EnvEval | Evaluated all tasks 0-{task_idx} at step 0")

        print(f"\n[Task {task_idx}] Starting Training")

        epoch_iter = trange(
            task_num_epochs,
            desc=f"Task {task_idx} Training",
            unit="epoch",
            disable=args.no_tqdm,
        )

        task_idx_arr = jnp.asarray(task_idx, dtype=jnp.int32)
        for epoch in epoch_iter:
            state, train_rng, mean_metrics, last_metrics = train_epoch_si(
                state,
                train_rng,
                train_obs,
                train_mean,
                train_logstd,
                args.batch_size,
                args.si_coeff,
                task_idx_arr,
            )

            mean_metrics_np = np.asarray(mean_metrics)
            last_metrics_np = np.asarray(last_metrics)
            global_step += train_steps_per_epoch
            epoch_iter.set_postfix(
                loss=f"{mean_metrics_np[0]:.4f}",
                dist=f"{mean_metrics_np[1]:.4f}",
                si=f"{mean_metrics_np[2]:.4f}",
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "Train_Loss/Total": float(mean_metrics_np[0]),
                        "Train_Loss/dist": float(mean_metrics_np[1]),
                        "Train_Loss/si_scaled": float(mean_metrics_np[2]),
                        "Train_Loss/si": float(mean_metrics_np[3]),
                        "Train_Loss/last_batch": float(last_metrics_np[0]),
                        "Training/epoch": epoch + 1,
                        "Training/global_updates": global_step,
                    },
                    step=global_step,
                )

            should_log_offline = ((epoch + 1) % offline_eval_every == 0) or (epoch + 1 == task_num_epochs)
            if should_log_offline:
                # Evaluate current task
                train_loss = float(
                    dataset_kl_loss_si(
                        state,
                        train_obs,
                        train_mean,
                        train_logstd,
                        task_idx_arr,
                    )
                )
                eval_msg = f"    Epoch {epoch + 1:04d} | Task {task_idx} | KL(train) = {train_loss:.6f}"
                if test_size > 0:
                    test_loss = float(
                        dataset_kl_loss_si(
                            state,
                            test_obs,
                            test_mean,
                            test_logstd,
                            task_idx_arr,
                        )
                    )
                    eval_msg += f" | KL(test) = {test_loss:.6f}"
                print(eval_msg)

                # Evaluate all tasks on offline data
                evaluate_all_tasks_offline(
                    state=state,
                    task_buffers=task_buffers,
                    current_task_idx=task_idx,
                    wandb_run=wandb_run,
                    global_step=global_step,
                    epoch=epoch + 1,
                )

            run_env_eval = args.env_eval_episodes > 0 and (
                ((epoch + 1) % env_eval_every == 0) or (epoch + 1 == task_num_epochs)
            )
            if run_env_eval:
                # Evaluate all tasks in environment
                evaluate_all_tasks_env(
                    state=state,
                    task_buffers=task_buffers,
                    current_task_idx=task_idx,
                    num_episodes=args.env_eval_episodes,
                    episode_length=task_data["episode_length"],
                    seed=args.seed + epoch,
                    wandb_run=wandb_run,
                    global_step=global_step,
                    epoch=epoch + 1,
                )
                print(f"      EnvEval | Evaluated all tasks 0-{task_idx}")

        state = consolidate_si_state(state, args.si_epsilon)
        print(f"Completed Task {task_idx}. SI buffers consolidated.")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Training complete in {total_time / 60.0:.2f} minutes.")
    print("=" * 80)

    if wandb_run is not None:
        wandb_run.log(
            {
                "Training/total_updates": total_updates_planned,
                "Training/elapsed_seconds": total_time,
                "Training/elapsed_minutes": total_time / 60.0,
            },
            step=global_step,
        )

    # Final evaluation
    print("\nFinal offline evaluation across all tasks:")
    for task_idx, task_data in enumerate(task_buffers):
        task_name = task_data["task_name"]
        test_size = task_data["test_size"]
        test_obs = task_data["test_obs"]
        test_mean = task_data["test_mean"]
        test_logstd = task_data["test_logstd"]
        if test_size == 0:
            print(f"  - {task_name}: no test split available.")
            continue
        loss = float(
            dataset_kl_loss_si(
                state,
                test_obs,
                test_mean,
                test_logstd,
                jnp.asarray(task_idx, dtype=jnp.int32),
            )
        )
        print(f"  - {task_name}: KL(test) = {loss:.6f}")

    if args.env_eval_episodes > 0:
        print("\nFinal environment evaluation:")
        for task_idx, task_data in enumerate(task_buffers):
            task_name = task_data["task_name"]
            env_metrics = evaluate_environment(
                task=task_data,
                state=state,
                task_idx=task_idx,
                task_name=task_name,
                num_episodes=args.env_eval_episodes,
                episode_length=task_data["episode_length"],
                seed=args.seed + 1234 + task_idx * 1000,
                wandb_run=wandb_run,
                log_to_wandb=wandb_run is not None,
                global_step=global_step,
                log_epoch=None,
            )
            print(
                f"  - {task_name}: "
                f"KL={env_metrics['env_loss']:.6f} "
                f"| TeacherR={env_metrics['teacher_return']:.3f} "
                f"| StudentR={env_metrics['student_return']:.3f} "
                f"| StudentSucc={env_metrics['student_success']:.3f}"
            )

    # Close all environments
    for task_data in task_buffers:
        task_data["env"].close()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
