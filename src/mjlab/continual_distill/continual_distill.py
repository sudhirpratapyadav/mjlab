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
# Load project-specific wandb credentials from .env file
from pathlib import Path as _Path
_env_file = _Path(__file__).parent / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()
    print(f"[continual_distill] Loaded wandb credentials from {_env_file}")

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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
    batch_weights: jnp.ndarray,
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
        # Weighted mean (weights normalized to mean 1.0 upstream; uniform weights
        # reduce this exactly to jnp.mean(kl_vals)).
        dist_loss = jnp.sum(batch_weights * kl_vals) / (jnp.sum(batch_weights) + 1e-8)
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


@functools.partial(jax.jit, static_argnums=(6,))
def train_epoch_si(
    state: StudentTrainStateSI,
    rng: jax.random.PRNGKey,
    obs: jnp.ndarray,
    teacher_mean: jnp.ndarray,
    teacher_logstd: jnp.ndarray,
    weights: jnp.ndarray,
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
    shuffled_w = jnp.take(weights, permutation, axis=0)

    batch_elems = num_batches * batch_size
    shuffled_obs = shuffled_obs[:batch_elems]
    shuffled_mean = shuffled_mean[:batch_elems]
    shuffled_logstd = shuffled_logstd[:batch_elems]
    shuffled_w = shuffled_w[:batch_elems]

    obs_batches = shuffled_obs.reshape((num_batches, batch_size, shuffled_obs.shape[-1]))
    mean_batches = shuffled_mean.reshape((num_batches, batch_size, shuffled_mean.shape[-1]))
    logstd_batches = shuffled_logstd.reshape((num_batches, batch_size, shuffled_logstd.shape[-1]))
    w_batches = shuffled_w.reshape((num_batches, batch_size))

    def batch_update(carry, batch):
        train_state, loss_sums = carry
        batch_obs, batch_mean, batch_logstd, batch_w = batch
        train_state, metrics = train_step_si(
            train_state,
            batch_obs,
            batch_mean,
            batch_logstd,
            batch_w,
            task_idx,
            si_coeff,
        )
        loss_sums = loss_sums + metrics
        return (train_state, loss_sums), metrics

    init_carry = (state, jnp.zeros(4, dtype=jnp.float32))
    (final_state, total_metrics), metrics_per_batch = jax.lax.scan(
        batch_update,
        init_carry,
        (obs_batches, mean_batches, logstd_batches, w_batches),
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
    # DEBUG: Start of evaluation
    print(f"\n{'='*80}")
    print(f"DEBUG EVAL START: Task {task_idx} ({task_name}) | Step {global_step} | Epoch {log_epoch}")
    print(f"{'='*80}")

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
    classical_teacher = task.get("classical_teacher")

    # DEBUG: Print environment state before evaluation
    print(f"Environment info:")
    print(f"  num_envs: {env.num_envs}")
    print(f"  max_episode_length: {env.max_episode_length}")
    print(f"  episode_length_buf: {env.episode_length_buf[:5].cpu().numpy()} (first 5)")

    # DEBUG: Check command manager state
    if hasattr(env, 'command_manager') and hasattr(env.command_manager, '_terms'):
        for cmd_name, cmd_term in env.command_manager._terms.items():
            print(f"  Command '{cmd_name}':")
            if hasattr(cmd_term, 'episode_success'):
                print(f"    episode_success: {cmd_term.episode_success[:5].cpu().numpy()} (first 5)")
            if hasattr(cmd_term, 'metrics'):
                for metric_name, metric_value in cmd_term.metrics.items():
                    if 'success' in metric_name.lower():
                        print(f"    metrics['{metric_name}']: {metric_value[:5].cpu().numpy()} (first 5)")

    task_idx_arr = jnp.asarray(task_idx, dtype=jnp.int32)

    # JIT compile action functions
    if classical_teacher is not None:

        def get_teacher_actions_batch(obs_jax):
            """Classical teacher: python policy on numpy observations."""
            return jnp.asarray(classical_teacher(np.asarray(obs_jax)))
    else:

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
    print(f"\n--- TEACHER ROLLOUT START ---")
    # Seed the reset so evaluation is reproducible and comparable across calls.
    # Without this, the env RNG drifts with accumulated stepping, so the same
    # weights get different goal poses at the periodic vs the final-sweep eval
    # (this produced the spurious low LiftCube "final" numbers).
    obs, _ = env.reset(seed=seed)
    if classical_teacher is not None:
        classical_teacher.reset()  # restart state machines with the episode
    num_envs = env.num_envs
    episode_returns = np.zeros(num_envs)
    teacher_total_kl = 0.0
    teacher_total_steps = 0

    print(f"After env.reset() (Teacher):")
    print(f"  num_envs: {num_envs}")
    print(f"  episode_length to run: {episode_length}")

    # Latch teacher success per-env across the rollout (same reason as student:
    # robust to mid-rollout auto-resets).
    teacher_cmd = env.command_manager.get_term(
        next(iter(env.command_manager.active_terms))
    )
    teacher_latched_success = np.zeros(num_envs)

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
        teacher_latched_success = np.maximum(
            teacher_latched_success, teacher_cmd.episode_success.detach().cpu().numpy()
        )

    teacher_episode_returns = episode_returns.tolist()
    teacher_successes = teacher_latched_success.tolist()
    print(f"  Teacher successes mean: {np.mean(teacher_successes):.4f}")
    print(f"--- TEACHER ROLLOUT END ---\n")

    # Student rollouts - parallel episodes (run full episode length, no early termination)
    print(f"\n--- STUDENT ROLLOUT START ---")
    # Same fixed seed as the teacher rollout above: teacher and student are then
    # scored on the SAME goal poses, and the eval is reproducible run-to-run.
    obs, _ = env.reset(seed=seed)
    episode_returns = np.zeros(num_envs)

    # DEBUG: Check command manager state AFTER reset
    print(f"After env.reset():")
    print(f"  episode_length_buf: {env.episode_length_buf[:5].cpu().numpy()} (first 5)")
    if hasattr(env, 'command_manager') and hasattr(env.command_manager, '_terms'):
        for cmd_name, cmd_term in env.command_manager._terms.items():
            if hasattr(cmd_term, 'episode_success'):
                print(f"  {cmd_name}.episode_success: {cmd_term.episode_success[:5].cpu().numpy()} (first 5)")
            if hasattr(cmd_term, 'target_pos'):
                print(f"  {cmd_name}.target_pos[0]: {cmd_term.target_pos[0].cpu().numpy()}")
            # Get object position if available
            if hasattr(cmd_term, 'object'):
                obj_pos = cmd_term.object.data.root_link_pos_w[0].cpu().numpy()
                print(f"  {cmd_name}.object_pos[0]: {obj_pos}")
                goal_error = np.linalg.norm(cmd_term.target_pos[0].cpu().numpy() - obj_pos)
                print(f"  {cmd_name}.initial_goal_error[0]: {goal_error:.4f}m")

    # Live per-env success, LATCHED every step. The env auto-resets on timeout,
    # which clears command_manager.episode_success; reading it once after the
    # loop (or worse, the pre-reduced scalar in info["log"]) therefore misses
    # successes and yields a stale/wrong number. np.maximum over the rollout
    # captures a success whenever it occurs. (This is the fix for the spurious
    # low LiftCube "final" accuracy — same weights read 1.0 vs 0.23 depending on
    # where the auto-reset boundary fell.)
    student_cmd = env.command_manager.get_term(
        next(iter(env.command_manager.active_terms))
    )
    latched_success = np.zeros(num_envs)

    # Run for full episode length (no early termination)
    for step in range(episode_length):
        obs_array = obs['policy'].cpu().numpy()
        obs_jax = jnp.array(obs_array)

        student_action_jax = get_student_actions_batch(obs_jax)
        student_action_torch = torch.from_numpy(np.array(student_action_jax))
        obs, reward, terminated, truncated, info = env.step(student_action_torch)
        episode_returns += reward.cpu().numpy()
        latched_success = np.maximum(
            latched_success, student_cmd.episode_success.detach().cpu().numpy()
        )

        # DEBUG: Log progress at certain steps
        if step in [0, episode_length//2, episode_length-1]:
            print(f"  Step {step}: episode_length_buf={env.episode_length_buf[0].item()}")
            if hasattr(env, 'command_manager') and hasattr(env.command_manager, '_terms'):
                for cmd_name, cmd_term in env.command_manager._terms.items():
                    if hasattr(cmd_term, 'episode_success'):
                        succ = cmd_term.episode_success[:5].cpu().numpy()
                        print(f"    {cmd_name}.episode_success[:5]: {succ}")

    # DEBUG: Check final step info
    print(f"\nAfter episode complete (step {episode_length}):")
    print(f"  episode_length_buf: {env.episode_length_buf[:5].cpu().numpy()} (first 5)")
    print(f"  terminated: {terminated[:5].cpu().numpy()} (first 5)")
    print(f"  truncated: {truncated[:5].cpu().numpy()} (first 5)")

    # DEBUG: Check command manager state BEFORE extracting from info
    if hasattr(env, 'command_manager') and hasattr(env.command_manager, '_terms'):
        for cmd_name, cmd_term in env.command_manager._terms.items():
            print(f"  {cmd_name} state BEFORE extraction:")
            if hasattr(cmd_term, 'episode_success'):
                print(f"    episode_success[:10]: {cmd_term.episode_success[:10].cpu().numpy()}")
            if hasattr(cmd_term, 'metrics'):
                for metric_name, metric_value in cmd_term.metrics.items():
                    if 'success' in metric_name.lower():
                        print(f"    metrics['{metric_name}'][:10]: {metric_value[:10].cpu().numpy()}")

    # Use the per-env success LATCHED across the rollout (see above). This is the
    # true per-episode success of THIS eval, robust to mid-rollout auto-resets —
    # unlike the old path that read a pre-reduced, episode-boundary-dependent
    # scalar from info["log"] and replicated it to all envs.
    student_episode_returns = episode_returns.tolist()
    student_successes = latched_success.tolist()

    print(f"  Final student_successes (latched): {student_successes[:10]} (first 10)")
    print(f"  Mean: {np.mean(student_successes):.4f}")
    print(f"--- STUDENT ROLLOUT END ---\n")

    # Compute metrics
    avg_env_kl = teacher_total_kl / max(teacher_total_steps, 1)
    metrics = {
        "env_loss": float(avg_env_kl),
        "teacher_return": float(np.mean(teacher_episode_returns)),
        "student_return": float(np.mean(student_episode_returns)),
        "teacher_success": float(np.mean(teacher_successes)),
        "student_success": float(np.mean(student_successes)),
    }

    # DEBUG: Print computed metrics
    print(f"\n{'='*80}")
    print(f"COMPUTED METRICS FOR TASK {task_idx} ({task_name}):")
    print(f"  env_loss: {metrics['env_loss']:.6f}")
    print(f"  teacher_return: {metrics['teacher_return']:.4f}")
    print(f"  student_return: {metrics['student_return']:.4f}")
    print(f"  teacher_success: {metrics['teacher_success']:.4f}")
    print(f"  student_success: {metrics['student_success']:.4f}")
    print(f"{'='*80}\n")

    if log_to_wandb and wandb_run is not None:
        payload = {
            f"EnvEval/task_{task_idx}_{task_name}/ts_loss/total": metrics["env_loss"],
            f"Rewards/task_{task_idx}_{task_name}": metrics["student_return"],
            f"Accuracy/task_{task_idx}_{task_name}": metrics["student_success"],
        }
        if log_epoch is not None:
            payload["Training/epoch"] = log_epoch
        # DEBUG: Print what's being logged to WandB
        print(f"LOGGING TO WANDB (step {global_step}):")
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print()
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

        # Evaluate on train split (using global student normalizer)
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
    print(f"\n{'#'*100}")
    print(f"### EVALUATE_ALL_TASKS_ENV CALLED ###")
    print(f"### Current training task: {current_task_idx}")
    print(f"### Global step: {global_step}, Epoch: {epoch}")
    print(f"### Will evaluate tasks 0 to {current_task_idx} (inclusive)")
    print(f"{'#'*100}\n")

    if num_episodes <= 0:
        return

    for eval_task_idx in range(len(task_buffers)):
        task_data = task_buffers[eval_task_idx]
        eval_task_name = task_data.get("task_name", f"task_{eval_task_idx}")

        if eval_task_idx > current_task_idx:
            # Log placeholder for future tasks
            print(f"  >>> Task {eval_task_idx} ({eval_task_name}): SKIPPED (future task)")
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

        print(f"\n  >>> Evaluating Task {eval_task_idx} ({eval_task_name}) in environment...")

        # Evaluate environment (using global student normalizer)
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

        print(f"  >>> Task {eval_task_idx} ({eval_task_name}) evaluation complete!")
        print(f"      Student success: {env_metrics['student_success']:.4f}")
        print()

    print(f"\n{'#'*100}")
    print(f"### EVALUATE_ALL_TASKS_ENV COMPLETE ###")
    print(f"{'#'*100}\n")


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


def _load_yaml_config(config_path: Path, task_sequence: List[str]) -> List[Dict[str, Any]]:
    """Load task configuration from YAML file and filter by sequence.

    Args:
        config_path: Path to tasks.yaml containing task definitions
        task_sequence: List of task names to load in order

    Returns:
        List of task configurations in the order specified by task_sequence
    """
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    # Load task definitions (new format: tasks as dict)
    if "tasks" in config:
        all_tasks = config["tasks"]
    # Support legacy format (task_info as list)
    elif "task_info" in config:
        all_tasks = {task["task_name"]: task for task in config["task_info"]}
    else:
        raise ValueError(f"No 'tasks' or 'task_info' entries found in {config_path}")

    # Build ordered list of tasks from sequence
    tasks = []
    for task_name in task_sequence:
        if task_name not in all_tasks:
            raise ValueError(f"Task '{task_name}' in sequence not found in {config_path}")

        task_def = all_tasks[task_name]
        # Add task_name to the dict if not present (for new format)
        if "task_name" not in task_def:
            task_def = dict(task_def)  # Make a copy
            task_def["task_name"] = task_name
        tasks.append(task_def)

    if not tasks:
        raise ValueError(f"No tasks specified in sequence: {task_sequence}")

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


def _build_global_normalizer(datasets: List[Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
    """Build global observation normalizer across all tasks.

    Aggregates statistics from all datasets to compute mean and std
    that will be used by the student for all tasks.

    Returns a dictionary with 'mean' and 'std' JAX arrays (JIT-compatible).
    """
    aggregate: Dict[str, Any] = {}
    obs_dim = None

    for data in datasets:
        obs = data["observations"]
        if obs.size == 0:
            continue
        obs_dim = obs.shape[1]
        batch_count = float(obs.shape[0])
        batch_mean = obs.mean(axis=0)
        batch_var = obs.var(axis=0)
        batch_summed_var = batch_var * batch_count

        if not aggregate:
            aggregate["count"] = batch_count
            aggregate["mean"] = batch_mean
            aggregate["summed_var"] = batch_summed_var
        else:
            count = aggregate["count"]
            mean = aggregate["mean"]
            summed_var = aggregate["summed_var"]
            total_count = count + batch_count
            delta = batch_mean - mean
            new_mean = mean + delta * (batch_count / total_count)
            m_a = summed_var
            m_b = batch_summed_var
            new_summed_var = m_a + m_b + (delta ** 2) * count * batch_count / total_count
            aggregate["count"] = total_count
            aggregate["mean"] = new_mean
            aggregate["summed_var"] = new_summed_var

    if not aggregate:
        if obs_dim is None:
            obs_dim = 1
        count = 1.0
        mean = np.zeros(obs_dim, dtype=np.float32)
        summed_var = np.ones(obs_dim, dtype=np.float32)
    else:
        count = aggregate["count"]
        mean = aggregate["mean"]
        summed_var = aggregate["summed_var"]

    std = np.sqrt(np.maximum(summed_var / max(count, 1.0), 1e-6)).astype(np.float32)

    return {
        'mean': jnp.array(mean),
        'std': jnp.array(std),
    }


def compute_distill_weights(
    teacher_mean: np.ndarray,
    teacher_logstd: np.ndarray,
    num_envs: int,
    mode: str,
    floor: float = 0.3,
    clip: float = 8.0,
) -> np.ndarray:
    """Per-sample distillation-loss weights (shape [N]), normalized to mean 1.0.

    IMPORTANT: the dataset is STEP-MAJOR. Collection stepped `num_envs` envs in
    parallel and appended one (num_envs)-row block per timestep, so
    obs[i] corresponds to (step = i // num_envs, env = i % num_envs). Temporal
    modes must reshape to [S, E, ...] with E = num_envs, NOT [n_ep, T].
    (Verified: with the correct layout, cube lifts off at step ~37 and |Δaction|
    is 0.31 during the grasp vs 0.007 in the hold — a 44x ratio; the wrong
    env-major reshape scrambled this to look flat.)

    Modes:
      uniform      : all 1.0 (baseline, == plain KL).
      delta_action : per-step |a_t - a_{t-1}|; grasp (high) up-weighted, hold (~0)
                     down-weighted, with a floor so hold samples don't vanish.
      perdim       : handled in-loss; returns uniform here.
    """
    n = teacher_mean.shape[0]
    E = int(num_envs)
    if mode in ("uniform", "perdim") or E <= 0 or n % E != 0:
        if mode == "delta_action" and (E <= 0 or n % E != 0):
            print(f"[weights] WARNING: N={n} not divisible by num_envs={E}; "
                  f"falling back to uniform weights.")
        return np.ones(n, dtype=np.float32)

    if mode == "delta_action":
        A = teacher_mean.shape[1]
        S = n // E
        m = teacher_mean.reshape(S, E, A)   # [step, env, action]
        d = np.zeros((S, E), dtype=np.float64)
        d[1:] = np.abs(m[1:] - m[:-1]).mean(axis=-1)
        d[0] = d[1]
        w = d.reshape(-1)
        # Weight = FLOOR + (|Δa| / median|Δa|), clipped. Using the MEDIAN (not max)
        # as the scale keeps the typical grasp step at a meaningful multiple while
        # rare outliers are clipped. FLOOR keeps the hold learnable. Then renorm to
        # mean 1.0 so LR/SI scale matches the uniform baseline.
        scale = np.median(w[w > 1e-6]) + 1e-8
        FLOOR = float(floor)
        CLIP = float(clip)
        w = FLOOR + np.clip(w / scale, 0.0, CLIP)
        w = w / (w.mean() + 1e-8)
        # report the grasp-vs-hold contrast actually applied
        we = w.reshape(S, E).mean(1)
        gr = float(we[:47].mean()); ho = float(we[60:150].mean()) if S >= 150 else float(we[60:].mean())
        print(f"[weights] delta_action (step-major, E={E}): min={w.min():.2f} "
              f"mean={w.mean():.2f} max={w.max():.2f} | grasp(t<47)={gr:.2f} "
              f"hold(t60+)={ho:.2f} ratio={gr/ho:.1f}x")
        return w.astype(np.float32)

    raise ValueError(f"Unknown distill weight mode: {mode}")


def _split_dataset_for_task(
    observations: np.ndarray,
    teacher_mean: np.ndarray,
    teacher_logstd: np.ndarray,
    train_fraction: float,
    episode_length: int,
    weights: np.ndarray = None,
) -> Tuple[np.ndarray, ...]:
    """Split dataset into train and test (optionally carrying per-sample weights)."""
    num_samples = observations.shape[0]
    if num_samples == 0:
        raise ValueError("Dataset is empty; cannot train.")
    if weights is None:
        weights = np.ones(num_samples, dtype=np.float32)
    train_fraction = float(np.clip(train_fraction, 0.0, 1.0))
    train_samples = int(train_fraction * num_samples)
    if episode_length > 0:
        train_samples = (train_samples // episode_length) * episode_length
    train_samples = max(min(train_samples, num_samples), 1)
    train_obs = observations[:train_samples]
    train_mean = teacher_mean[:train_samples]
    train_logstd = teacher_logstd[:train_samples]
    train_w = weights[:train_samples]
    test_obs = observations[train_samples:]
    test_mean = teacher_mean[train_samples:]
    test_logstd = teacher_logstd[train_samples:]
    return train_obs, train_mean, train_logstd, train_w, test_obs, test_mean, test_logstd


def save_checkpoint(
    state: StudentTrainStateSI,
    checkpoint_dir: Path,
    task_idx: int,
    epoch: int,
    global_step: int,
    task_sequence: List[str],
    env_ids: List[str],
    obs_dim: int,
    action_dim: int,
    hidden_dims: Tuple[int, ...],
) -> None:
    """Save student checkpoint.

    Args:
        state: Student training state
        checkpoint_dir: Directory to save checkpoint
        task_idx: Current task index
        epoch: Current epoch within task
        global_step: Global training step
        task_sequence: List of task names in order
        env_ids: List of environment IDs in order
        obs_dim: Observation dimension
        action_dim: Action dimension
        hidden_dims: Hidden layer dimensions
    """
    checkpoint = {
        # Network weights
        "params": state.params,
        # Normalizer (dict with 'mean' and 'std' JAX arrays)
        "normalizer_params": state.normalizer_params,
        # Task sequence mapping (list of task names and env_ids)
        "task_sequence": task_sequence,
        "env_ids": env_ids,
        # Network architecture info
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "num_tasks": state.num_tasks,
        "hidden_dims": hidden_dims,
        "student_min_std": state.student_min_std,
        # Training info
        "task_idx": task_idx,
        "epoch": epoch,
        "global_step": global_step,
    }

    # Convert JAX arrays to numpy for serialization
    def jax_to_numpy(tree):
        """Convert JAX arrays in pytree to numpy arrays."""
        import jax.tree_util as tree_util
        return tree_util.tree_map(lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, tree)

    checkpoint_np = jax_to_numpy(checkpoint)

    # Save checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_task{task_idx}_epoch{epoch}_step{global_step}.pkl"
    with checkpoint_path.open("wb") as f:
        pickle.dump(checkpoint_np, f)

    print(f"    Saved checkpoint: {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Continual policy distillation with Synaptic Intelligence (SI) for mjlab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tasks-config", type=Path, required=True, help="YAML file containing task definitions (tasks.yaml).")
    parser.add_argument("--task-sequence", type=str, nargs="+", required=True, help="Sequence of task names to train on (space-separated).")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size for SGD/Adam updates.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of each dataset used for training.")
    parser.add_argument("--student-min-std", type=float, default=1e-3, help="Minimum std for student policy outputs.")
    parser.add_argument("--student-hidden-dims", type=int, nargs="+", default=[4096, 2048, 1024], help="Hidden layer sizes of the student MLP.")
    parser.add_argument("--distill-weight-mode", type=str, default="uniform", choices=["uniform", "delta_action", "perdim"], help="Per-sample distillation-loss weighting (uniform=plain KL; delta_action=up-weight high action-change/grasp steps).")
    parser.add_argument("--distill-weight-floor", type=float, default=0.3, help="delta_action: min weight for low-action-change (hold) samples.")
    parser.add_argument("--distill-weight-clip", type=float, default=8.0, help="delta_action: max weight multiple (in median-scaled units) before floor.")
    parser.add_argument("--si-coeff", type=float, default=1.0, help="Regularization coefficient for SI surrogate.")
    parser.add_argument("--si-epsilon", type=float, default=1e-3, help="Stability term for SI consolidation.")
    parser.add_argument("--eval-every", type=int, default=20, help="Frequency (in epochs) of offline evaluation.")
    parser.add_argument("--env-eval-every", type=int, default=100, help="Frequency (in epochs) of environment evaluation; 0 => match --eval-every.")
    parser.add_argument("--env-eval-episodes", type=int, default=64, help="Number of episodes for environment evaluation.")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Frequency (in epochs) to save checkpoints.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Base directory to save checkpoints (default: ./results in same folder as script).")
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

    task_configs = _load_yaml_config(args.tasks_config, args.task_sequence)

    # Load datasets and teachers
    datasets: List[Dict[str, Any]] = []
    teacher_infos: List[Dict[str, Any]] = []
    for task in task_configs:
        dataset_folder = Path(task["dataset_folder"]).expanduser()
        data, teacher_info = _load_dataset_and_teacher(dataset_folder)
        datasets.append(data)
        teacher_infos.append(teacher_info)

    # Build global normalizer for student (shared across all tasks)
    global_normalizer = _build_global_normalizer(datasets)
    print(f"Built global observation normalizer across {len(datasets)} tasks")

    # Determine device
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get dimensions from environment (not YAML config)
    print("\nExtracting dimensions from environment...")
    first_env_id = task_configs[0]["env_id"]
    temp_env_cfg = load_env_cfg(first_env_id, test=True)
    temp_env_cfg.scene.num_envs = 1
    temp_env_cfg.seed = args.seed
    temp_env = ManagerBasedRlEnv(cfg=temp_env_cfg, device=device)

    obs_sample, _ = temp_env.reset()
    obs_dim = obs_sample['policy'].shape[-1]
    action_dim = temp_env.action_space.shape[-1]
    temp_env.close()

    print(f"  Observation dimension: {obs_dim}")
    print(f"  Action dimension: {action_dim}")

    # Validate all tasks have the same dimensions
    print("Validating dimensions across all tasks...")
    for task_idx, task_cfg in enumerate(task_configs):
        temp_cfg = load_env_cfg(task_cfg["env_id"], test=True)
        temp_cfg.scene.num_envs = 1
        temp_cfg.seed = args.seed
        temp = ManagerBasedRlEnv(cfg=temp_cfg, device=device)
        obs_s, _ = temp.reset()
        task_obs_dim = obs_s['policy'].shape[-1]
        task_action_dim = temp.action_space.shape[-1]
        temp.close()

        if task_obs_dim != obs_dim:
            raise ValueError(f"Task {task_idx} ({task_cfg['env_id']}) has obs_dim={task_obs_dim}, expected {obs_dim}")
        if task_action_dim != action_dim:
            raise ValueError(f"Task {task_idx} ({task_cfg['env_id']}) has action_dim={task_action_dim}, expected {action_dim}")

    print(f"✓ All tasks validated with obs_dim={obs_dim}, action_dim={action_dim}")

    num_tasks = len(task_configs)

    # Create student policy
    student = StudentPolicy(
        obs_size=obs_dim,
        action_size=action_dim,
        num_tasks=num_tasks,
        hidden_dims=tuple(args.student_hidden_dims),
    )
    init_key = jax.random.PRNGKey(args.seed)
    params = student.init(init_key)
    flat_params = student.flatten_tree(params)
    optimizer = optax.adam(args.learning_rate)

    # Initialize state with global normalizer (shared across all tasks)
    state = StudentTrainStateSI.create(
        apply_fn=student.network.apply,
        params=params,
        tx=optimizer,
        normalizer_params=global_normalizer,
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

    for task_idx, task_cfg in enumerate(task_configs):
        dataset = datasets[task_idx]
        observations = dataset["observations"]
        action_targets = dataset["action_targets"]
        teacher_mean = action_targets[:, :action_dim]
        teacher_logstd = action_targets[:, action_dim:]

        task_name = task_cfg.get("task_name", f"task_{task_idx}")
        task_slug = _slugify(task_name) or f"task_{task_idx}"
        dataset_folder_str = str(Path(task_cfg["dataset_folder"]).expanduser())
        num_epochs = int(task_cfg.get("num_epochs", 500))  # Default to 500 if not specified in YAML

        # Create mjlab environment first to get actual episode length
        env_id = task_cfg["env_id"]
        env_cfg = load_env_cfg(env_id, test=True)
        env_cfg.scene.num_envs = args.env_eval_episodes
        env_cfg.seed = args.seed
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

        # Use environment's actual episode length
        episode_length = env.max_episode_length
        print(f"[INFO] Task {task_idx} ({task_name}): Using environment episode length: {episode_length} steps")

        # Per-sample distillation weights (A4 etc.); uniform == baseline KL.
        # Dataset is STEP-MAJOR: reshape needs the collection num_envs (from
        # metadata), not the eval episode_length.
        collect_num_envs = int(dataset.get("metadata", {}).get("num_envs", 0))
        sample_weights = compute_distill_weights(
            teacher_mean, teacher_logstd, collect_num_envs,
            args.distill_weight_mode,
            floor=args.distill_weight_floor,
            clip=args.distill_weight_clip,
        )

        # Split dataset using actual episode length
        train_obs_np, train_mean_np, train_logstd_np, train_w_np, test_obs_np, test_mean_np, test_logstd_np = _split_dataset_for_task(
            observations,
            teacher_mean,
            teacher_logstd,
            args.train_fraction,
            episode_length,
            sample_weights,
        )

        train_size = int(train_obs_np.shape[0])
        test_size = int(test_obs_np.shape[0])
        if train_size < args.batch_size:
            raise ValueError(
                f"Task {task_idx} training set ({train_size}) smaller than batch size ({args.batch_size})."
            )
        train_steps_per_epoch = max(1, train_size // args.batch_size)

        # Load teacher from preloaded teacher_info (already in JAX format!)
        teacher_info = teacher_infos[task_idx]
        print(f"[Task {task_idx}] Loading teacher from: {dataset_folder_str}/teacher.pkl")

        # Create teacher policy and normalizer from loaded info.
        # Classical (scripted) teachers carry no JAX weights; EnvEval rolls
        # them out via the python policy object instead.
        classical_teacher = None
        if teacher_info.get('teacher_type') == 'classical':
            from mjlab.continual_distill import classical as classical_mod

            classical_cls = getattr(classical_mod, teacher_info['classical_class'])
            classical_teacher = classical_cls(num_envs=env.num_envs)
            teacher_policy = None
            teacher_params = None
        else:
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

        task_buffers.append(
            {
                "train_obs": jnp.asarray(train_obs_np),
                "train_mean": jnp.asarray(train_mean_np),
                "train_logstd": jnp.asarray(train_logstd_np),
                "train_weights": jnp.asarray(train_w_np),
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
                "env": env,
                "episode_length": episode_length,
                "teacher_policy": teacher_policy,
                "teacher_params": teacher_params,
                "teacher_std": teacher_std,
                "teacher_normalizer": teacher_normalizer,
                "classical_teacher": classical_teacher,
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

    # Create checkpoint directory structure: base_dir/{run_name}/
    # Default base_dir is ./results in the same folder as this script
    if args.checkpoint_dir is None:
        script_dir = Path(__file__).parent
        base_checkpoint_dir = script_dir / "results"
    else:
        base_checkpoint_dir = args.checkpoint_dir

    checkpoint_dir = base_checkpoint_dir / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    total_updates_planned = sum(tb["num_epochs"] * tb["train_steps_per_epoch"] for tb in task_buffers)
    run_config = {
        "num_tasks": num_tasks,
        "tasks_config": str(args.tasks_config),
        "task_sequence": args.task_sequence,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "student_min_std": args.student_min_std,
        "student_hidden_dims": list(args.student_hidden_dims),
        "distill_weight_mode": args.distill_weight_mode,
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

        train_obs = task_data["train_obs"]
        train_mean = task_data["train_mean"]
        train_logstd = task_data["train_logstd"]
        train_weights = task_data["train_weights"]
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
                train_weights,
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

            # Save checkpoint
            should_save_checkpoint = ((epoch + 1) % args.checkpoint_every == 0) or (epoch + 1 == task_num_epochs)
            if should_save_checkpoint:
                # Build env_ids list from task_configs
                env_ids = [tc["env_id"] for tc in task_configs]
                save_checkpoint(
                    state=state,
                    checkpoint_dir=checkpoint_dir,
                    task_idx=task_idx,
                    epoch=epoch + 1,
                    global_step=global_step,
                    task_sequence=args.task_sequence,
                    env_ids=env_ids,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dims=tuple(args.student_hidden_dims),
                )

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
        # The last periodic eval ran at epoch index (num_epochs - 1) of the last
        # task, with seed = args.seed + epoch (see evaluate_all_tasks_env call).
        # Reuse that epoch so this final sweep reproduces it exactly.
        last_epoch = task_buffers[-1]["num_epochs"] - 1
        for task_idx, task_data in enumerate(task_buffers):
            task_name = task_data["task_name"]

            env_metrics = evaluate_environment(
                task=task_data,
                state=state,
                task_idx=task_idx,
                task_name=task_name,
                num_episodes=args.env_eval_episodes,
                episode_length=task_data["episode_length"],
                # Use the SAME seed the periodic eval used at the final epoch of
                # the last task, so this final sweep reproduces that eval instead
                # of scoring on a different (uncontrolled) set of goal poses.
                seed=(args.seed + last_epoch) + task_idx * 1000,
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
