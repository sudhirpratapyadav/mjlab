#!/usr/bin/env python3
"""
Simple policy distillation for mjlab environments.

This script performs standard distillation of a single teacher policy into a student policy.
The student uses the same architecture as the teacher but is initialized randomly.

Key features:
- Single task distillation (no continual learning)
- Student uses same architecture as teacher
- Tracks both teacher and student success rates during evaluation
- No Synaptic Intelligence (SI) regularization
"""

import argparse
import functools
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Tuple

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
from flax.training import train_state
from tqdm import trange

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.continual_distill.utils import (
    ObservationNormalizer,
    TeacherActorMLP,
    TeacherPolicy,
)

try:
    import wandb
except ImportError:
    wandb = None


class StudentTrainState(train_state.TrainState):
    """Training state for student policy."""
    normalizer_params: Any
    action_dim: int
    student_min_std: float


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
        print("[distill] wandb not installed; disabling tracking.")
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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get student distribution parameters.

    Args:
        apply_fn: Network apply function
        params: Network parameters
        normalizer_params: Observation normalizer dict with 'mean' and 'std'
        obs: Observations
        student_min_std: Minimum std for student policy
        action_dim: Action dimension

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

    # Split into mean and scale parameters
    loc, scale_params = jnp.split(logits, 2, axis=-1)
    std = jax.nn.softplus(scale_params) + student_min_std
    log_std = jnp.log(std)

    return loc, log_std


@jax.jit
def train_step(
    state: StudentTrainState,
    batch_obs: jnp.ndarray,
    batch_teacher_mean: jnp.ndarray,
    batch_teacher_logstd: jnp.ndarray,
) -> Tuple[StudentTrainState, jnp.ndarray]:
    """Single training step.

    Args:
        state: Training state
        batch_obs: Batch of observations
        batch_teacher_mean: Teacher means
        batch_teacher_logstd: Teacher logstds

    Returns:
        Tuple of (updated state, loss)
    """
    def loss_fn(params):
        student_mean, student_logstd = compute_student_distribution(
            state.apply_fn,
            params,
            state.normalizer_params,
            batch_obs,
            state.student_min_std,
            state.action_dim,
        )
        kl_vals = gaussian_kl(
            batch_teacher_mean, batch_teacher_logstd, student_mean, student_logstd
        )
        return jnp.mean(kl_vals)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


@functools.partial(jax.jit, static_argnums=(5,))
def train_epoch(
    state: StudentTrainState,
    rng: jax.random.PRNGKey,
    obs: jnp.ndarray,
    teacher_mean: jnp.ndarray,
    teacher_logstd: jnp.ndarray,
    batch_size: int,
) -> Tuple[StudentTrainState, jax.random.PRNGKey, jnp.ndarray, jnp.ndarray]:
    """Train for one epoch.

    Args:
        state: Training state
        rng: Random key
        obs: All observations
        teacher_mean: All teacher means
        teacher_logstd: All teacher logstds
        batch_size: Mini-batch size (static)

    Returns:
        Tuple of (updated state, updated rng, mean loss, last batch loss)
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
        train_state, loss_sum = carry
        batch_obs, batch_mean, batch_logstd = batch
        train_state, loss = train_step(
            train_state,
            batch_obs,
            batch_mean,
            batch_logstd,
        )
        loss_sum = loss_sum + loss
        return (train_state, loss_sum), loss

    init_carry = (state, jnp.array(0.0, dtype=jnp.float32))
    (final_state, total_loss), losses_per_batch = jax.lax.scan(
        batch_update,
        init_carry,
        (obs_batches, mean_batches, logstd_batches),
    )

    mean_loss = total_loss / num_batches
    last_loss = losses_per_batch[-1]
    return final_state, rng, mean_loss, last_loss


@jax.jit
def dataset_kl_loss(
    state: StudentTrainState,
    obs: jnp.ndarray,
    teacher_mean: jnp.ndarray,
    teacher_logstd: jnp.ndarray,
) -> jnp.ndarray:
    """Compute KL loss on entire dataset."""
    mean_pred, logstd_pred = compute_student_distribution(
        state.apply_fn,
        state.params,
        state.normalizer_params,
        obs,
        state.student_min_std,
        state.action_dim,
    )
    kl_values = gaussian_kl(teacher_mean, teacher_logstd, mean_pred, logstd_pred)
    return jnp.mean(kl_values)


def evaluate_environment(
    env,
    state: StudentTrainState,
    teacher_policy: TeacherPolicy,
    teacher_params,
    teacher_std,
    teacher_normalizer,
    num_episodes: int,
    episode_length: int,
    seed: int,
    wandb_run=None,
    log_to_wandb: bool = False,
    global_step: int = None,
    log_epoch: int = None,
) -> Dict[str, float]:
    """Evaluate student and teacher policies in mjlab environment.

    Args:
        env: mjlab environment
        state: Student training state
        teacher_policy: Teacher policy
        teacher_params: Teacher parameters
        teacher_std: Teacher action std
        teacher_normalizer: Teacher observation normalizer
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
            "env_kl": 0.0,
            "teacher_return": 0.0,
            "student_return": 0.0,
            "teacher_success": 0.0,
            "student_success": 0.0,
        }

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
        )
        return mean

    # Teacher rollouts - parallel episodes (run full episode length, no early termination)
    print(f"\n[DEBUG] Starting teacher evaluation: {num_episodes} episodes in parallel, {episode_length} steps")

    obs, _ = env.reset()
    num_envs = env.num_envs
    episode_returns = np.zeros(num_envs)
    teacher_total_kl = 0.0
    teacher_total_steps = 0

    # Run for full episode length (no early termination)
    for step in range(episode_length):
        obs_array = obs['policy'].cpu().numpy()
        obs_jax = jnp.array(obs_array)

        # Get actions and compute KL
        teacher_action_jax = get_teacher_actions_batch(obs_jax)
        student_action_jax = get_student_actions_batch(obs_jax)

        teacher_std_jax = jnp.array(teacher_std)
        teacher_logstd = jnp.log(teacher_std_jax)
        _, student_logstd = compute_student_distribution(
            state.apply_fn, state.params, state.normalizer_params,
            obs_jax, state.student_min_std, state.action_dim,
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
    print(f"[DEBUG] Teacher: returns={teacher_episode_returns}, successes={teacher_successes}")

    # Student rollouts - parallel episodes (run full episode length, no early termination)
    print(f"\n[DEBUG] Starting student evaluation: {num_episodes} episodes in parallel, {episode_length} steps")

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
    print(f"[DEBUG] Student: returns={student_episode_returns}, successes={student_successes}")

    # Compute metrics
    avg_env_kl = teacher_total_kl / max(teacher_total_steps, 1)

    metrics = {
        "env_kl": float(avg_env_kl),
        "teacher_return": float(np.mean(teacher_episode_returns)),
        "student_return": float(np.mean(student_episode_returns)),
        "teacher_success": float(np.mean(teacher_successes)),
        "student_success": float(np.mean(student_successes)),
    }

    if log_to_wandb and wandb_run is not None:
        payload = {
            "EnvEval/kl": metrics["env_kl"],
            "EnvEval/teacher_return": metrics["teacher_return"],
            "EnvEval/student_return": metrics["student_return"],
            "EnvEval/teacher_success": metrics["teacher_success"],
            "EnvEval/student_success": metrics["student_success"],
        }
        if log_epoch is not None:
            payload["Training/epoch"] = log_epoch
        wandb_run.log(payload, step=global_step)

    return metrics


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load task configuration from YAML file and return first task."""
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    tasks = config.get("task_info", [])
    if not tasks:
        raise ValueError(f"No task_info entries found in {config_path}")
    if len(tasks) == 0:
        raise ValueError(f"No tasks found in {config_path}")
    # Return only the first task
    return tasks[0]


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


def _build_normalizer(observations: np.ndarray) -> Dict[str, jnp.ndarray]:
    """Build observation normalizer parameters.

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


def _split_dataset(
    observations: np.ndarray,
    teacher_mean: np.ndarray,
    teacher_logstd: np.ndarray,
    train_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into train and test."""
    num_samples = observations.shape[0]
    if num_samples == 0:
        raise ValueError("Dataset is empty; cannot train.")
    train_fraction = float(np.clip(train_fraction, 0.0, 1.0))
    train_samples = int(train_fraction * num_samples)
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
        description="Simple policy distillation for mjlab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML file describing the task (uses first task only)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size for SGD/Adam updates")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of dataset used for training")
    parser.add_argument("--student-min-std", type=float, default=1e-3, help="Minimum std for student policy outputs")
    parser.add_argument("--eval-every", type=int, default=50, help="Frequency (in epochs) of evaluation")
    parser.add_argument("--env-eval-episodes", type=int, default=4, help="Number of episodes for environment evaluation")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--track", action="store_true", default=True, help="Enable Weights & Biases logging")
    parser.add_argument("--no-track", action="store_false", dest="track", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="distill_mjlab", help="wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity/team")
    parser.add_argument("--wandb-mode", type=str, default="online", help="wandb mode (online/offline/disabled)")
    parser.add_argument("--run-name", type=str, default=None, help="Optional wandb run name")
    parser.add_argument("--device", type=str, default=None, help="Device for mjlab environments (default: cuda:0 if available)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load YAML config and extract first task
    task_config = _load_yaml_config(args.config)

    # Extract task parameters
    dataset_folder = Path(task_config["dataset_folder"]).expanduser()
    env_id = task_config["env_id"]
    obs_dim = int(task_config["obs_dim"])
    action_dim = int(task_config["action_dim"])
    episode_length = int(task_config.get("ep_len", 1000))
    num_epochs = int(task_config.get("num_epochs", 500))
    task_name = task_config.get("task_name", "task_0")

    # Load dataset and teacher
    dataset, teacher_info = _load_dataset_and_teacher(dataset_folder)

    observations = dataset["observations"]
    action_targets = dataset["action_targets"]

    teacher_mean = action_targets[:, :action_dim]
    teacher_logstd = action_targets[:, action_dim:]

    # Build observation normalizer
    normalizer = _build_normalizer(observations)

    # Split dataset
    train_obs_np, train_mean_np, train_logstd_np, test_obs_np, test_mean_np, test_logstd_np = _split_dataset(
        observations,
        teacher_mean,
        teacher_logstd,
        args.train_fraction,
    )

    train_obs = jnp.asarray(train_obs_np)
    train_mean = jnp.asarray(train_mean_np)
    train_logstd = jnp.asarray(train_logstd_np)
    test_obs = jnp.asarray(test_obs_np)
    test_mean = jnp.asarray(test_mean_np)
    test_logstd = jnp.asarray(test_logstd_np)

    train_size = int(train_obs_np.shape[0])
    test_size = int(test_obs_np.shape[0])
    if train_size < args.batch_size:
        raise ValueError(
            f"Training set ({train_size}) smaller than batch size ({args.batch_size})."
        )
    train_steps_per_epoch = max(1, train_size // args.batch_size)

    print("=" * 80)
    print("Policy Distillation - mjlab")
    print("=" * 80)
    print(f"Task name: {task_name}")
    print(f"Dataset folder: {dataset_folder}")
    print(f"Environment: {env_id}")
    print(f"Obs dim: {obs_dim} | Action dim: {action_dim}")
    print(f"Train samples: {train_size} | Test samples: {test_size}")
    print(f"Learning rate: {args.learning_rate} | Batch size: {args.batch_size}")
    print(f"Num epochs: {num_epochs} | Episode length: {episode_length}")

    # Create teacher policy from loaded info
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

    # Create student policy with SAME architecture as teacher but random initialization
    # Student outputs 2*action_dim (mean + scale_params)
    student_network = TeacherActorMLP(
        action_size=2 * action_dim,  # Output mean + scale parameters
        hidden_dims=tuple(teacher_info['hidden_dims']),
    )

    init_key = jax.random.PRNGKey(args.seed)
    dummy_obs = jnp.zeros((1, obs_dim))
    student_params = student_network.init(init_key, dummy_obs)

    optimizer = optax.adam(args.learning_rate)

    state = StudentTrainState.create(
        apply_fn=student_network.apply,
        params=student_params,
        tx=optimizer,
        normalizer_params=normalizer,
        action_dim=action_dim,
        student_min_std=args.student_min_std,
    )

    # Create mjlab environment for evaluation
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(env_id, test=True)
    env_cfg.scene.num_envs = args.env_eval_episodes  # Run episodes in parallel
    env_cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    # Use environment's actual episode length for evaluation
    env_episode_length = env.max_episode_length
    print(f"[INFO] Using environment episode length: {env_episode_length} steps (overriding config ep_len={episode_length})")
    print(f"[INFO] Running {args.env_eval_episodes} episodes in parallel")
    episode_length = env_episode_length

    args.run_name = args.run_name or f"distill_{task_name}_{int(time.time())}"

    run_config = {
        "task_name": task_name,
        "dataset_folder": str(dataset_folder),
        "env_id": env_id,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "train_size": train_size,
        "test_size": test_size,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "student_min_std": args.student_min_std,
        "train_fraction": args.train_fraction,
        "num_epochs": num_epochs,
        "eval_every": args.eval_every,
        "env_eval_episodes": args.env_eval_episodes,
        "episode_length": episode_length,
    }
    wandb_run = init_wandb(args, run_config)

    start_time = time.time()
    train_rng = jax.random.PRNGKey(args.seed + 1)
    global_step = 0

    # Initial evaluation at step 0 (before training)
    print("\n" + "=" * 80)
    print("Initial Evaluation (Step 0 - Before Training)")
    print("=" * 80)

    if args.env_eval_episodes > 0:
        env_metrics = evaluate_environment(
            env=env,
            state=state,
            teacher_policy=teacher_policy,
            teacher_params=teacher_params,
            teacher_std=teacher_std,
            teacher_normalizer=teacher_normalizer,
            num_episodes=args.env_eval_episodes,
            episode_length=episode_length,
            seed=args.seed,
            wandb_run=wandb_run,
            log_to_wandb=wandb_run is not None,
            global_step=global_step,
            log_epoch=0,
        )
        print(
            f"  Step 0 EnvEval | KL={env_metrics['env_kl']:.6f} "
            f"| Teacher: R={env_metrics['teacher_return']:.3f}, Acc={env_metrics['teacher_success']:.3f} "
            f"| Student: R={env_metrics['student_return']:.3f}, Acc={env_metrics['student_success']:.3f}"
        )

    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    epoch_iter = trange(
        num_epochs,
        desc="Training",
        unit="epoch",
        disable=args.no_tqdm,
    )

    for epoch in epoch_iter:
        state, train_rng, mean_loss, last_loss = train_epoch(
            state,
            train_rng,
            train_obs,
            train_mean,
            train_logstd,
            args.batch_size,
        )

        mean_loss_np = float(np.asarray(mean_loss))
        last_loss_np = float(np.asarray(last_loss))
        global_step += train_steps_per_epoch

        epoch_iter.set_postfix(
            loss=f"{mean_loss_np:.4f}",
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "Train/loss": mean_loss_np,
                    "Train/last_batch_loss": last_loss_np,
                    "Training/epoch": epoch + 1,
                    "Training/global_step": global_step,
                },
                step=global_step,
            )

        should_eval = ((epoch + 1) % args.eval_every == 0) or (epoch + 1 == num_epochs)
        if should_eval:
            # Evaluate on train split
            train_loss = float(
                dataset_kl_loss(
                    state,
                    train_obs,
                    train_mean,
                    train_logstd,
                )
            )
            eval_msg = f"    Epoch {epoch + 1:04d} | KL(train) = {train_loss:.6f}"

            # Evaluate on test split
            if test_size > 0:
                test_loss = float(
                    dataset_kl_loss(
                        state,
                        test_obs,
                        test_mean,
                        test_logstd,
                    )
                )
                eval_msg += f" | KL(test) = {test_loss:.6f}"

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "Eval/train_kl": train_loss,
                            "Eval/test_kl": test_loss,
                            "Training/epoch": epoch + 1,
                        },
                        step=global_step,
                    )
            else:
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "Eval/train_kl": train_loss,
                            "Training/epoch": epoch + 1,
                        },
                        step=global_step,
                    )

            print(eval_msg)

            # Environment evaluation
            if args.env_eval_episodes > 0:
                env_metrics = evaluate_environment(
                    env=env,
                    state=state,
                    teacher_policy=teacher_policy,
                    teacher_params=teacher_params,
                    teacher_std=teacher_std,
                    teacher_normalizer=teacher_normalizer,
                    num_episodes=args.env_eval_episodes,
                    episode_length=episode_length,
                    seed=args.seed + epoch,
                    wandb_run=wandb_run,
                    log_to_wandb=wandb_run is not None,
                    global_step=global_step,
                    log_epoch=epoch + 1,
                )
                print(
                    f"      EnvEval | KL={env_metrics['env_kl']:.6f} "
                    f"| Teacher: R={env_metrics['teacher_return']:.3f}, Acc={env_metrics['teacher_success']:.3f} "
                    f"| Student: R={env_metrics['student_return']:.3f}, Acc={env_metrics['student_success']:.3f}"
                )

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Training complete in {total_time / 60.0:.2f} minutes.")
    print("=" * 80)

    if wandb_run is not None:
        wandb_run.log(
            {
                "Training/elapsed_seconds": total_time,
                "Training/elapsed_minutes": total_time / 60.0,
            },
            step=global_step,
        )

    # Final evaluation
    print("\nFinal evaluation:")

    # Offline evaluation
    if test_size > 0:
        test_loss = float(
            dataset_kl_loss(
                state,
                test_obs,
                test_mean,
                test_logstd,
            )
        )
        print(f"  Test KL loss: {test_loss:.6f}")

    # Environment evaluation
    if args.env_eval_episodes > 0:
        env_metrics = evaluate_environment(
            env=env,
            state=state,
            teacher_policy=teacher_policy,
            teacher_params=teacher_params,
            teacher_std=teacher_std,
            teacher_normalizer=teacher_normalizer,
            num_episodes=args.env_eval_episodes,
            episode_length=episode_length,
            seed=args.seed + 1234,
            wandb_run=wandb_run,
            log_to_wandb=wandb_run is not None,
            global_step=global_step,
            log_epoch=None,
        )
        print(
            f"  Environment evaluation:\n"
            f"    KL: {env_metrics['env_kl']:.6f}\n"
            f"    Teacher - Return: {env_metrics['teacher_return']:.3f}, Success: {env_metrics['teacher_success']:.3f}\n"
            f"    Student - Return: {env_metrics['student_return']:.3f}, Success: {env_metrics['student_success']:.3f}"
        )

    # Close environment
    env.close()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
