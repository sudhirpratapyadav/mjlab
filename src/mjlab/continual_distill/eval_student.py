#!/usr/bin/env python3
"""
Evaluate student policy in environment with viser viewer.

Loads a student checkpoint, runs test episodes, and launches viser viewer
for interactive visualization.

Usage:
    python eval_student.py --checkpoint /path/to/checkpoint.pkl --env-id Isaac-Lift-Cube-Franka-v0
"""

import argparse
import os
import pickle
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jnp
import numpy as np
import torch
import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.continual_distill.utils import StudentPolicy


class EnvWrapper:
    """Wrapper to add get_observations method for viewer compatibility."""

    def __init__(self, env):
        self.env = env
        self._last_obs = None

    def get_observations(self):
        """Return last observations from the environment."""
        if self._last_obs is None:
            # If no observations yet, reset the environment
            self._last_obs, _ = self.env.reset()
        return self._last_obs

    def step(self, actions):
        """Step the environment and cache observations."""
        obs, reward, terminated, truncated, info = self.env.step(actions)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def reset(self):
        """Reset the environment and cache observations."""
        obs, info = self.env.reset()
        self._last_obs = obs
        return obs, info

    def close(self):
        """Close the environment."""
        self.env.close()

    def __getattr__(self, name):
        """Delegate all other attributes to wrapped env."""
        return getattr(self.env, name)


class StudentPolicyWrapper:
    """Wrapper to make student JAX policy compatible with mjlab viewer."""

    def __init__(self, student_policy, params, normalizer_params, task_idx, student_min_std, action_dim):
        self.student_policy = student_policy
        self.params = params
        self.normalizer_params = normalizer_params
        self.task_idx = jnp.asarray(task_idx, dtype=jnp.int32)
        self.student_min_std = student_min_std
        self.action_dim = action_dim

        # JIT compile the action function
        @jax.jit
        def get_action_jit(obs_jax, task_idx):
            # Normalize observations
            if isinstance(normalizer_params, dict):
                normalized_obs = (obs_jax - normalizer_params['mean']) / (normalizer_params['std'] + 1e-8)
            else:
                normalized_obs = normalizer_params.normalize(obs_jax)

            # Forward pass through network
            logits = student_policy.network.apply(params, normalized_obs)

            # Extract head for this task
            head_dim = 2 * action_dim
            start = task_idx * head_dim
            head_logits = jax.lax.dynamic_slice_in_dim(logits, start_index=start, slice_size=head_dim, axis=-1)

            # Split into mean and scale parameters
            loc, scale_params = jnp.split(head_logits, 2, axis=-1)

            return loc

        self.get_action_jit = get_action_jit

    def __call__(self, obs_dict):
        """Get action from observation (compatible with viewer interface)."""
        # Get observation as numpy array
        obs_array = obs_dict['policy'].cpu().numpy()

        # Convert to JAX array
        obs_jax = jnp.array(obs_array)

        # Get action using JAX network
        action_jax = self.get_action_jit(obs_jax, self.task_idx)

        # Convert action back to PyTorch tensor for environment
        action_torch = torch.from_numpy(np.array(action_jax))

        return action_torch


def load_checkpoint(checkpoint_path: Path):
    """Load student checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pkl file

    Returns:
        Dictionary containing checkpoint data
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    with checkpoint_path.open("rb") as f:
        checkpoint = pickle.load(f)

    # Convert numpy arrays back to JAX
    def numpy_to_jax(tree):
        import jax.tree_util as tree_util
        return tree_util.tree_map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, tree)

    checkpoint['params'] = numpy_to_jax(checkpoint['params'])

    # Handle normalizer_params (dict with 'mean' and 'std')
    if isinstance(checkpoint['normalizer_params'], dict):
        checkpoint['normalizer_params'] = {
            'mean': jnp.array(checkpoint['normalizer_params']['mean']),
            'std': jnp.array(checkpoint['normalizer_params']['std']),
        }
    else:
        checkpoint['normalizer_params'] = numpy_to_jax(checkpoint['normalizer_params'])

    return checkpoint


def load_env_ids_from_yaml(task_sequence, yaml_path=None):
    """Load env_ids from tasks.yaml config file.

    Args:
        task_sequence: List of task names from checkpoint
        yaml_path: Optional path to tasks.yaml file. If None, uses default location.

    Returns:
        List of env_ids corresponding to task_sequence, or None if not found
    """
    if yaml_path is None:
        # Default to config/tasks.yaml relative to this script
        script_dir = Path(__file__).parent
        yaml_path = script_dir / "config" / "tasks.yaml"

    if not yaml_path.exists():
        return None

    try:
        with yaml_path.open("r") as f:
            config = yaml.safe_load(f)

        # Load task definitions (new format: tasks as dict)
        if "tasks" in config:
            all_tasks = config["tasks"]
        # Support legacy format (task_info as list)
        elif "task_info" in config:
            all_tasks = {task["task_name"]: task for task in config["task_info"]}
        else:
            return None

        # Build env_ids list from task_sequence
        env_ids = []
        for task_name in task_sequence:
            if task_name in all_tasks:
                env_id = all_tasks[task_name].get("env_id")
                if env_id:
                    env_ids.append(env_id)
                else:
                    return None  # Missing env_id
            else:
                return None  # Task not found in yaml

        return env_ids

    except Exception as e:
        print(f"Warning: Could not load tasks.yaml: {e}")
        return None


def test_student(
    env,
    student_policy: StudentPolicy,
    params,
    normalizer_params,
    task_idx: int,
    student_min_std: float,
    action_dim: int,
    num_episodes: int,
    episode_length: int,
):
    """Run student policy in environment and print results.

    Args:
        env: mjlab environment
        student_policy: Student policy instance
        params: Student parameters
        normalizer_params: Observation normalizer
        task_idx: Task index to use (head selection)
        student_min_std: Minimum std for student policy
        action_dim: Action dimension
        num_episodes: Number of episodes to run
        episode_length: Maximum steps per episode
    """
    task_idx_arr = jnp.asarray(task_idx, dtype=jnp.int32)

    @jax.jit
    def get_student_action(obs_jax):
        """Get student action for batch of observations."""
        # Normalize observations
        if isinstance(normalizer_params, dict):
            normalized_obs = (obs_jax - normalizer_params['mean']) / (normalizer_params['std'] + 1e-8)
        else:
            normalized_obs = normalizer_params.normalize(obs_jax)

        # Forward pass through network
        logits = student_policy.network.apply(params, normalized_obs)

        # Extract head for this task
        head_dim = 2 * action_dim
        start = task_idx_arr * head_dim
        head_logits = jax.lax.dynamic_slice_in_dim(logits, start_index=start, slice_size=head_dim, axis=-1)

        # Split into mean and scale parameters
        loc, _ = jnp.split(head_logits, 2, axis=-1)

        return loc

    episode_returns = []
    episode_successes = []

    print(f"\nRunning {num_episodes} test episodes (max {episode_length} steps each)...")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        steps = 0
        success_val = 0.0

        for step in range(episode_length):
            # Get observation and convert to JAX
            obs_array = obs['policy'].cpu().numpy()
            obs_jax = jnp.array(obs_array)

            # Get student action
            action_jax = get_student_action(obs_jax)

            # Step environment
            action_torch = torch.from_numpy(np.array(action_jax))
            obs, reward, terminated, truncated, info = env.step(action_torch)

            episode_return += reward.cpu().numpy().mean()
            steps += 1

            # Extract success from info["log"] when episode terminates
            if (terminated.any() or truncated.any()) and "log" in info and isinstance(info["log"], dict):
                for key, value in info["log"].items():
                    if "episode_success" in key.lower():
                        success_val = float(value)
                        break
                # Episode terminated, exit loop
                break

        episode_returns.append(episode_return)
        episode_successes.append(success_val)

        # Print episode summary
        success_str = "SUCCESS" if success_val > 0.5 else "FAIL"
        print(f"Episode {ep + 1:3d}: Steps={steps:3d} Return={episode_return:7.2f} Success={success_val:.2f} [{success_str}]")

    # Print summary
    print("=" * 60)
    print(f"Average Return:  {np.mean(episode_returns):.3f} ± {np.std(episode_returns):.3f}")
    print(f"Success Rate:    {np.mean(episode_successes):.3f} ({int(np.sum(episode_successes))}/{num_episodes})")
    print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate student policy in environment with viser viewer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to student checkpoint .pkl file",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        required=True,
        help="Environment ID (e.g., Isaac-Lift-Cube-Franka-v0) - will automatically determine task index from checkpoint",
    )
    parser.add_argument(
        "--task-idx",
        type=int,
        default=None,
        help="Task index (0-based) to evaluate - optional, will be auto-determined from env-id if not provided",
    )
    parser.add_argument(
        "--tasks-config",
        type=Path,
        default=None,
        help="Path to tasks.yaml config file (default: ./config/tasks.yaml relative to script)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments (viewer requires 1)",
    )
    parser.add_argument(
        "--num-test-episodes",
        type=int,
        default=5,
        help="Number of test episodes to run before viewer",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda:0 if available)",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test episodes and go directly to viewer",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load checkpoint
    checkpoint_path = args.checkpoint.expanduser()
    print("=" * 60)
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)

    obs_dim = checkpoint['obs_dim']
    action_dim = checkpoint['action_dim']
    num_tasks = checkpoint['num_tasks']
    hidden_dims = tuple(checkpoint['hidden_dims'])
    student_min_std = checkpoint['student_min_std']
    task_sequence = checkpoint['task_sequence']
    env_ids = checkpoint.get('env_ids', [])

    print(f"Checkpoint info:")
    print(f"  Observation dimension: {obs_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Number of tasks: {num_tasks}")
    print(f"  Hidden dimensions: {hidden_dims}")
    print(f"  Trained up to task: {checkpoint['task_idx']} (epoch {checkpoint['epoch']}, step {checkpoint['global_step']})")
    print("=" * 60)

    # Display task sequence
    print(f"\nTask sequence in checkpoint:")
    for idx, task_name in enumerate(task_sequence):
        env_id_str = f" ({env_ids[idx]})" if idx < len(env_ids) else ""
        print(f"  Task {idx}: {task_name}{env_id_str}")

    # Determine task index
    if args.task_idx is not None:
        # Use provided task index
        task_idx = args.task_idx
        if task_idx < 0 or task_idx >= num_tasks:
            print(f"Error: Task index {task_idx} is out of range. Must be between 0 and {num_tasks-1}")
            return
        print(f"\nUsing provided task index {task_idx}: {task_sequence[task_idx]}")
    else:
        # Auto-determine from env_id
        if not env_ids:
            print("\nCheckpoint does not contain env_ids. Attempting to load from tasks.yaml...")
            env_ids = load_env_ids_from_yaml(task_sequence, yaml_path=args.tasks_config)

            if env_ids:
                print(f"Successfully loaded env_ids from tasks.yaml:")
                for idx, (task_name, env_id) in enumerate(zip(task_sequence, env_ids)):
                    print(f"  Task {idx}: {task_name} ({env_id})")
            else:
                print("Error: Could not load env_ids from tasks.yaml. Please provide --task-idx manually.")
                print(f"  Tried loading from: {args.tasks_config or 'default location (./config/tasks.yaml)'}")
                return

        if args.env_id not in env_ids:
            print(f"Error: Environment ID '{args.env_id}' not found in checkpoint.")
            print(f"Available env_ids: {env_ids}")
            return

        task_idx = env_ids.index(args.env_id)
        print(f"\nAuto-determined task index {task_idx} from env-id '{args.env_id}'")
        print(f"Task name: {task_sequence[task_idx]}")

    print(f"Environment: {args.env_id}")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create student policy
    student_policy = StudentPolicy(
        obs_size=obs_dim,
        action_size=action_dim,
        num_tasks=num_tasks,
        hidden_dims=hidden_dims,
    )

    params = checkpoint['params']
    normalizer_params = checkpoint['normalizer_params']

    # Run test episodes first (unless skipped)
    if not args.skip_test:
        print(f"\nRunning {args.num_test_episodes} test episodes first...")
        print(f"Creating environment: {args.env_id} on {device}")

        # Create environment for testing
        env_cfg_test = load_env_cfg(args.env_id, test=True)
        env_cfg_test.scene.num_envs = 1
        env_cfg_test.seed = args.seed
        env_test = ManagerBasedRlEnv(cfg=env_cfg_test, device=device)

        # Run test episodes
        test_student(
            env=env_test,
            student_policy=student_policy,
            params=params,
            normalizer_params=normalizer_params,
            task_idx=task_idx,
            student_min_std=student_min_std,
            action_dim=action_dim,
            num_episodes=args.num_test_episodes,
            episode_length=args.episode_length,
        )

        # Close test environment
        env_test.close()

    # Create environment for viewer (needs num_envs=1)
    print(f"\nCreating environment for viewer: {args.env_id} on {device}")
    env_cfg = load_env_cfg(args.env_id, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_base = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    # Wrap environment for viewer compatibility
    env = EnvWrapper(env_base)

    # Create policy wrapper for viewer
    policy = StudentPolicyWrapper(
        student_policy=student_policy,
        params=params,
        normalizer_params=normalizer_params,
        task_idx=task_idx,
        student_min_std=student_min_std,
        action_dim=action_dim,
    )

    # Start viser viewer
    print("\n" + "=" * 60)
    print("Starting Viser Viewer")
    print("=" * 60)
    print("Open browser to view the visualization")
    print("The viewer URL will be displayed below")
    print("=" * 60)

    from mjlab.viewer import ViserPlayViewer

    # Run viewer
    ViserPlayViewer(env, policy).run()

    # Cleanup
    env.close()
    print("\nViewer closed!")


if __name__ == "__main__":
    main()
