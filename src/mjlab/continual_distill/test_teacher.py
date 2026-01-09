#!/usr/bin/env python3
"""
Test teacher policy in environment with viser viewer.

Loads a teacher policy, runs test episodes to print success/reward stats,
then launches viser viewer for interactive visualization.

Usage:
    python test_teacher.py --dataset-folder /path/to/dataset --env-id Isaac-Lift-Cube-Franka-v0
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

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.continual_distill.utils import ObservationNormalizer, TeacherPolicy


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


class JAXPolicyWrapper:
    """Wrapper to make JAX policy compatible with mjlab viewer."""

    def __init__(self, teacher_policy, teacher_params, teacher_normalizer):
        self.teacher_policy = teacher_policy
        self.teacher_params = teacher_params
        self.teacher_normalizer = teacher_normalizer

        # JIT compile the action function
        @jax.jit
        def get_action_jit(obs_jax):
            normalized_obs = teacher_normalizer.normalize(obs_jax)
            action_mean = teacher_policy.actor.apply(teacher_params, normalized_obs)
            return action_mean

        self.get_action_jit = get_action_jit

    def __call__(self, obs_dict):
        """Get action from observation (compatible with viewer interface)."""
        # Get observation as numpy array
        obs_array = obs_dict['policy'].cpu().numpy()

        # Convert to JAX array
        obs_jax = jnp.array(obs_array)

        # Get action using JAX network
        action_jax = self.get_action_jit(obs_jax)

        # Convert action back to PyTorch tensor for environment
        action_torch = torch.from_numpy(np.array(action_jax))

        return action_torch


def load_teacher(dataset_folder: Path):
    """Load teacher policy from dataset folder.

    Args:
        dataset_folder: Path to folder containing teacher.pkl

    Returns:
        Tuple of (teacher_policy, teacher_params, teacher_std, teacher_normalizer, obs_dim, action_dim)
    """
    teacher_path = dataset_folder / "teacher.pkl"
    if not teacher_path.exists():
        raise FileNotFoundError(f"teacher.pkl not found in {dataset_folder}")

    # Load teacher
    with teacher_path.open("rb") as f:
        teacher_info = pickle.load(f)

    # Convert numpy to JAX arrays
    def numpy_to_jax(tree):
        import jax.tree_util as tree_util
        return tree_util.tree_map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, tree)

    teacher_info['jax_params'] = numpy_to_jax(teacher_info['jax_params'])
    teacher_info['action_std'] = jnp.array(teacher_info['action_std'])
    teacher_info['obs_normalizer_mean'] = jnp.array(teacher_info['obs_normalizer_mean'])
    teacher_info['obs_normalizer_std'] = jnp.array(teacher_info['obs_normalizer_std'])

    # Get dimensions
    obs_dim = teacher_info['obs_dim']
    action_dim = teacher_info['action_dim']

    # Create teacher policy
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

    return teacher_policy, teacher_params, teacher_std, teacher_normalizer, obs_dim, action_dim


def test_teacher(
    env,
    teacher_policy: TeacherPolicy,
    teacher_params,
    teacher_normalizer,
    num_episodes: int,
    episode_length: int,
):
    """Run teacher policy in environment and print results.

    Args:
        env: mjlab environment
        teacher_policy: Teacher policy instance
        teacher_params: Teacher parameters
        teacher_normalizer: Observation normalizer
        num_episodes: Number of episodes to run
        episode_length: Maximum steps per episode
    """
    @jax.jit
    def get_teacher_action(obs_jax):
        """Get teacher action for batch of observations."""
        normalized_obs = teacher_normalizer.normalize(obs_jax)
        action_mean = teacher_policy.actor.apply(teacher_params, normalized_obs)
        return action_mean

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

            # Get teacher action
            action_jax = get_teacher_action(obs_jax)

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
        description="Test teacher policy in environment with viser viewer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-folder",
        type=Path,
        required=True,
        help="Folder containing teacher.pkl",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        required=True,
        help="Environment ID (e.g., Isaac-Lift-Cube-Franka-v0)",
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

    # Load teacher
    dataset_folder = args.dataset_folder.expanduser()
    print("=" * 60)
    print(f"Loading teacher from: {dataset_folder}")
    teacher_policy, teacher_params, teacher_std, teacher_normalizer, obs_dim, action_dim = load_teacher(
        dataset_folder
    )
    print(f"Teacher loaded: obs_dim={obs_dim}, action_dim={action_dim}")
    print("=" * 60)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Run test episodes first (unless skipped)
    if not args.skip_test:
        print(f"\nRunning {args.num_test_episodes} test episodes first...")
        print(f"Creating environment: {args.env_id} on {device}")

        # Create environment for testing (use test=True for no corruption but proper episode length)
        env_cfg_test = load_env_cfg(args.env_id, test=True)
        env_cfg_test.scene.num_envs = 1
        env_cfg_test.seed = args.seed
        env_test = ManagerBasedRlEnv(cfg=env_cfg_test, device=device)

        # Run test episodes
        test_teacher(
            env=env_test,
            teacher_policy=teacher_policy,
            teacher_params=teacher_params,
            teacher_normalizer=teacher_normalizer,
            num_episodes=args.num_test_episodes,
            episode_length=args.episode_length,
        )

        # Close test environment
        env_test.close()

    # Create environment for viewer (needs num_envs=1)
    print(f"\nCreating environment for viewer: {args.env_id} on {device}")
    env_cfg = load_env_cfg(args.env_id, play=True)
    env_cfg.scene.num_envs = 1  # Viewer requires single environment
    env_cfg.seed = args.seed
    env_base = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    # Wrap environment for viewer compatibility
    env = EnvWrapper(env_base)

    # Create policy wrapper for viewer
    policy = JAXPolicyWrapper(teacher_policy, teacher_params, teacher_normalizer)

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
