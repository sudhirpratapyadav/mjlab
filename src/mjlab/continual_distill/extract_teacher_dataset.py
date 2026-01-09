#!/usr/bin/env python3
"""
Extract Teacher Policy Dataset for Policy Distillation from mjlab checkpoints.

This script collects state-action pairs from trained teacher policies in mjlab environments.

Key Features:
- Loads teacher policy from PyTorch checkpoint and converts to JAX
- Uses vectorized environments for fast parallel data collection
- Collects observations (state vectors) and actions (mean + std)
- Saves dataset as pickle file with metadata

Usage:
    python extract_teacher_dataset.py \\
        --checkpoint /path/to/model.pt \\
        --task Mjlab-Open-Door-Franka \\
        --num-samples 150000 \\
        --num-envs 512 \\
        --output-dir teacher_data

    Note: Episode length is automatically extracted from the environment test configuration.
"""

import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import torch
from tqdm import tqdm

# Import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

# Import our utilities
from mjlab.continual_distill.utils import load_teacher_from_checkpoint


def collect_dataset(
    teacher,
    actor_params: dict,
    action_std: jnp.ndarray,
    obs_normalizer,
    env,
    num_samples: int,
    num_envs: int,
    episode_length: int,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Collect dataset from teacher policy rollouts using vectorized environments.

    Args:
        teacher: TeacherPolicy instance
        actor_params: Actor network parameters
        action_std: Action standard deviation
        obs_normalizer: Observation normalizer
        env: mjlab environment
        num_samples: Target number of total samples to collect
        num_envs: Number of parallel environments
        episode_length: Maximum steps per episode
        seed: Random seed

    Returns:
        Dictionary with observations, action_targets (mean + logstd)
    """
    print(f"\n{'='*80}")
    print(f"Collecting ~{num_samples} samples from {num_envs} parallel envs")
    print(f"{'='*80}")

    # JIT compile the action function for speed
    # Vectorize over batch dimension (num_envs)
    @jax.jit
    def get_actions_batch(params, obs_batch):
        """Get action mean for a batch of observations."""
        # obs_batch: [num_envs, obs_dim]
        normalized_obs = obs_normalizer.normalize(obs_batch)
        action_means = teacher.actor.apply(params, normalized_obs)
        return action_means

    # Calculate number of episodes needed
    samples_per_episode = num_envs * episode_length
    num_episodes = (num_samples + samples_per_episode - 1) // samples_per_episode

    print(f"  Samples per episode: {samples_per_episode}")
    print(f"  Number of episodes: {num_episodes}")
    print(f"  Episode length: {episode_length}")

    # Storage for dataset
    all_observations = []
    all_action_means = []
    all_action_stds = []

    # Reset environment
    obs, _ = env.reset()

    # Collect data with progress bar
    total_steps = num_episodes * episode_length
    with tqdm(total=total_steps, desc="Collecting data") as pbar:
        for episode_idx in range(num_episodes):
            # Reset at start of each episode
            if episode_idx > 0:
                obs, _ = env.reset()

            for step_idx in range(episode_length):
                # Get observations as JAX array
                obs_array = obs['policy'].cpu().numpy()
                obs_jax = jnp.array(obs_array)

                # Get action means from teacher (vectorized)
                action_means_jax = get_actions_batch(actor_params, obs_jax)

                # Store data (convert to numpy)
                obs_np = np.array(obs_jax)
                action_means_np = np.array(action_means_jax)

                # Expand action_std to match batch size
                action_stds_np = np.tile(np.array(action_std), (num_envs, 1))

                all_observations.append(obs_np)
                all_action_means.append(action_means_np)
                all_action_stds.append(action_stds_np)

                # Step environment with action means (deterministic)
                action_torch = torch.from_numpy(action_means_np)
                obs, reward, terminated, truncated, info = env.step(action_torch)

                pbar.update(1)

    # Concatenate and reshape data
    observations = np.concatenate(all_observations, axis=0)  # [total_steps, obs_dim]
    action_means = np.concatenate(all_action_means, axis=0)  # [total_steps, action_dim]
    action_stds = np.concatenate(all_action_stds, axis=0)    # [total_steps, action_dim]

    # Compute log_std from std
    action_logstds = np.log(action_stds + 1e-8)

    # Concatenate mean and logstd to match distillation format
    # action_targets shape: (N, action_dim*2) where first half is mean, second half is logstd
    action_targets = np.concatenate([action_means, action_logstds], axis=1)

    # Print statistics
    print(f"\n{'='*80}")
    print("Collection Statistics")
    print(f"{'='*80}")
    print(f"Total samples collected: {len(observations)}")
    print(f"Observation shape: {observations.shape}")
    print(f"Action targets shape: {action_targets.shape}")
    print(f"  Action means range: [{action_means.min():.3f}, {action_means.max():.3f}]")
    print(f"  Action stds range: [{action_stds.min():.3f}, {action_stds.max():.3f}]")

    return {
        'observations': observations,
        'action_targets': action_targets,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract teacher policy dataset for policy distillation from mjlab checkpoints',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to teacher policy PyTorch checkpoint (.pt file)'
    )
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        help='mjlab task ID (e.g., Mjlab-Open-Door-Franka)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=150000,
        help='Target number of samples to collect'
    )
    parser.add_argument(
        '--num-envs',
        type=int,
        default=512,
        help='Number of parallel environments'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for dataset pickle files (default: continual_distill/teacher_datasets)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for environment'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device for environment (default: cuda:0 if available, else cpu)'
    )

    args = parser.parse_args()

    print("="*80)
    print("mjlab Teacher Dataset Extraction")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Num envs: {args.num_envs}")
    print(f"Target samples: {args.num_samples}")

    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint).resolve()  # Get absolute path
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print(f"Checkpoint: {checkpoint_path}")

    # Create output directory
    if args.output_dir is None:
        # Default: continual_distill/teacher_datasets
        script_dir = Path(__file__).parent
        output_dir = script_dir / 'teacher_datasets'
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    # Load mjlab environment
    print(f"\n{'='*80}")
    print("Loading mjlab Environment")
    print(f"{'='*80}")

    env_cfg = load_env_cfg(args.task, test=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    # Get episode length from environment configuration
    episode_length = env.max_episode_length
    print(f"✓ Environment loaded")
    print(f"  Device: {device}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Episode length: {episode_length} steps")

    # Get dimensions
    obs_sample, _ = env.reset()
    obs_size = obs_sample['policy'].shape[-1]
    action_size = env.action_space.shape[-1]

    print(f"  Observation size: {obs_size}")
    print(f"  Action size: {action_size}")

    # Load teacher policy
    print(f"\n{'='*80}")
    print("Loading Teacher Policy")
    print(f"{'='*80}")

    teacher, actor_params, action_std, obs_normalizer = load_teacher_from_checkpoint(
        str(checkpoint_path),
        obs_size,
        action_size,
    )

    print(f"✓ Teacher policy loaded")
    print(f"  Hidden dims: {teacher.hidden_dims}")
    print(f"  Action std shape: {action_std.shape}")
    print(f"  Action std values: {action_std}")

    # Collect dataset
    dataset = collect_dataset(
        teacher=teacher,
        actor_params=actor_params,
        action_std=action_std,
        obs_normalizer=obs_normalizer,
        env=env,
        num_samples=args.num_samples,
        num_envs=args.num_envs,
        episode_length=episode_length,
        seed=args.seed,
    )

    # Clean up environment
    env.close()

    # Create output folder structure
    checkpoint_name = checkpoint_path.stem  # e.g., "model_100"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name_slug = args.task.replace('-', '_')

    # Folder name: {task}_{checkpoint}_{timestamp}
    folder_name = f"{task_name_slug}_{checkpoint_name}_{timestamp}"
    task_folder = output_dir / folder_name
    task_folder.mkdir(parents=True, exist_ok=True)

    # Convert JAX parameters to numpy for pickling
    def jax_to_numpy(tree):
        """Convert JAX arrays in pytree to numpy arrays."""
        import jax.tree_util as tree_util
        return tree_util.tree_map(lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, tree)

    teacher_params_numpy = jax_to_numpy(actor_params)

    # ===== Save Teacher Weights =====
    teacher_data = {
        'jax_params': teacher_params_numpy,  # Already converted PyTorch → JAX
        'action_std': np.array(action_std),
        'obs_normalizer_mean': np.array(obs_normalizer.mean),
        'obs_normalizer_std': np.array(obs_normalizer.std),
        'hidden_dims': teacher.hidden_dims,  # Network architecture (e.g., (512, 256, 128))

        # Metadata
        'checkpoint_path': str(checkpoint_path),  # Original PyTorch checkpoint (for reference)
        'checkpoint_name': checkpoint_name,
        'task': args.task,
        'obs_dim': obs_size,
        'action_dim': action_size,
    }

    teacher_path = task_folder / 'teacher.pkl'
    with open(teacher_path, 'wb') as f:
        pickle.dump(teacher_data, f)

    # ===== Save Dataset (observations + action targets) =====
    dataset_with_metadata = {
        'observations': dataset['observations'],
        'action_targets': dataset['action_targets'],

        # Basic metadata
        'metadata': {
            'task': args.task,
            'obs_dim': obs_size,
            'action_dim': action_size,
            'num_samples_requested': args.num_samples,
            'num_envs': args.num_envs,
            'episode_length': episode_length,
            'total_samples': len(dataset['observations']),
            'collection_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'seed': args.seed,
            'device': device,
        }
    }

    data_path = task_folder / 'data.pkl'
    with open(data_path, 'wb') as f:
        pickle.dump(dataset_with_metadata, f)

    print(f"\n{'='*80}")
    print(f"Saving Dataset & Teacher")
    print(f"{'='*80}")
    print(f"Output folder: {task_folder}")
    print(f"  ├─ data.pkl     ({dataset['observations'].shape[0]} samples)")
    print(f"  └─ teacher.pkl  (JAX weights + normalizer)")

    print(f"\n✓ Dataset saved successfully!")

    # Print summary
    print(f"\n{'='*80}")
    print("Dataset Summary")
    print(f"{'='*80}")
    print(f"  Task: {args.task}")
    print(f"  Observations shape: {dataset['observations'].shape}")
    print(f"  Action targets shape: {dataset['action_targets'].shape}")
    print(f"  Total samples: {len(dataset['observations'])}")
    print(f"  Checkpoint: {checkpoint_name}")
    print(f"  Original checkpoint path: {checkpoint_path}")
    print(f"  Teacher network: JAX format, {teacher.hidden_dims}")
    print(f"  Output folder: {folder_name}")
    print(f"  Full path: {task_folder}")

    # Print usage example
    print(f"\n{'='*80}")
    print("Usage Example")
    print(f"{'='*80}")
    print(f"In YAML config, use:")
    print(f"")
    print(f"task_info:")
    print(f"  - task_name: \"{args.task.split('-')[-2].title() if '-' in args.task else args.task}\"")
    print(f"    env_id: \"{args.task}\"")
    print(f"    dataset_folder: \"{folder_name}\"")
    print(f"    num_epochs: 500")
    print(f"")
    print(f"To load in Python:")
    print(f"")
    print(f"import pickle")
    print(f"from pathlib import Path")
    print(f"")
    print(f"folder = Path('{task_folder}')")
    print(f"")
    print(f"# Load dataset")
    print(f"with open(folder / 'data.pkl', 'rb') as f:")
    print(f"    data = pickle.load(f)")
    print(f"observations = data['observations']  # Shape: {dataset['observations'].shape}")
    print(f"action_targets = data['action_targets']  # Shape: {dataset['action_targets'].shape}")
    print(f"")
    print(f"# Load teacher (JAX format, no conversion needed!)")
    print(f"with open(folder / 'teacher.pkl', 'rb') as f:")
    print(f"    teacher = pickle.load(f)")
    print(f"teacher_params = teacher['jax_params']")
    print(f"teacher_std = teacher['action_std']")
    print(f"obs_mean = teacher['obs_normalizer_mean']")
    print(f"obs_std = teacher['obs_normalizer_std']")
    print(f"hidden_dims = teacher['hidden_dims']")


if __name__ == "__main__":
    main()
