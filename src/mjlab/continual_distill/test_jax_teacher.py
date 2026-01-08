#!/usr/bin/env python3
"""
Test script to verify JAX teacher network can load mjlab PyTorch checkpoints
and run with mjlab environments.

This script:
1. Defines a JAX teacher network matching mjlab's ActorCritic structure
2. Loads PyTorch checkpoint and converts weights to JAX format
3. Runs the mjlab environment with the JAX network for a few episodes
4. Prints basic statistics to verify everything works

Usage:
    python test_jax_teacher.py --checkpoint /path/to/model.pt --task franka-lift-cube-v0
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, Callable

os.environ["JAX_PLATFORMS"] = "cpu"  # Use CPU for testing

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import linen as nn


# ============================================================================
# JAX Teacher Network (matches mjlab ActorCritic structure)
# ============================================================================

class TeacherActorMLP(nn.Module):
    """
    JAX MLP matching mjlab's actor network structure.

    Architecture:
    - Hidden layers with ELU activation
    - Output layer for action mean
    - Separate parameter for action std (not state-dependent by default)
    """
    action_size: int
    hidden_dims: Tuple[int, ...] = (512, 256, 128)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through actor MLP.

        Args:
            obs: Normalized observations [batch_size, obs_dim]

        Returns:
            action_mean: Mean of action distribution [batch_size, action_size]
        """
        x = obs

        # Hidden layers with ELU activation
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(
                hidden_dim,
                name=f'fc_{i}',
                kernel_init=nn.initializers.lecun_uniform(),
            )(x)
            x = nn.elu(x)

        # Output layer (action mean)
        action_mean = nn.Dense(
            self.action_size,
            name=f'fc_{len(self.hidden_dims)}',
            kernel_init=nn.initializers.lecun_uniform(),
        )(x)

        return action_mean


class ObservationNormalizer:
    """Running statistics for observation normalization (matches PyTorch EmpiricalNormalization)."""

    def __init__(self, mean: jnp.ndarray, std: jnp.ndarray):
        self.mean = mean
        self.std = std

    def normalize(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Normalize observations using running statistics."""
        return (obs - self.mean) / (self.std + 1e-8)


class TeacherPolicy:
    """
    Teacher policy wrapper for mjlab actor network.

    This matches the mjlab actor network:
    - Actor MLP for action mean
    - Separate std parameter (scalar or per-action)
    - Observation normalization
    - Normal distribution (no tanh)

    Note: We only load the policy/actor network, not the critic.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
    ):
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_dims = hidden_dims

        # Create actor network
        self.actor = TeacherActorMLP(
            action_size=action_size,
            hidden_dims=hidden_dims,
        )

    def init(self, key: jax.random.PRNGKey) -> dict:
        """Initialize network parameters."""
        dummy_obs = jnp.zeros((1, self.obs_size))
        params = self.actor.init(key, dummy_obs)
        return params

    def get_action(
        self,
        actor_params: dict,
        action_std: jnp.ndarray,
        obs_normalizer: ObservationNormalizer,
        obs: jnp.ndarray,
        key: jax.random.PRNGKey,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Get action from observation.

        Args:
            actor_params: Actor network parameters
            action_std: Standard deviation for actions
            obs_normalizer: Observation normalizer
            obs: Raw observations
            key: JAX random key
            deterministic: If True, return mean; else sample

        Returns:
            action: Sampled or deterministic action
        """
        # Normalize observation
        normalized_obs = obs_normalizer.normalize(obs)

        # Get action mean from actor network
        action_mean = self.actor.apply(actor_params, normalized_obs)

        if deterministic:
            return action_mean
        else:
            # Sample from Normal distribution
            noise = jax.random.normal(key, shape=action_mean.shape)
            return action_mean + action_std * noise


# ============================================================================
# PyTorch Checkpoint Loading & Conversion
# ============================================================================

def load_pytorch_checkpoint(checkpoint_path: str):
    """Load PyTorch checkpoint from mjlab training.

    Returns:
        checkpoint: Full checkpoint dict containing model_state_dict, optimizer, etc.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def convert_pytorch_to_jax_params(
    pytorch_state_dict: dict,
    obs_size: int,
    action_size: int,
    hidden_dims: Tuple[int, ...],
) -> Tuple[dict, jnp.ndarray, ObservationNormalizer]:
    """Convert PyTorch actor network parameters to JAX format.

    We only extract the actor/policy network, not the critic.

    PyTorch structure (from rsl_rl ActorCritic actor):
    - actor.0.weight, actor.0.bias  (first hidden layer)
    - actor.2.weight, actor.2.bias  (second hidden layer)
    - actor.4.weight, actor.4.bias  (third hidden layer)
    - actor.6.weight, actor.6.bias  (output layer)
    - std or log_std (action noise parameter)
    - actor_obs_normalizer.mean, actor_obs_normalizer.std, etc.

    JAX structure (Flax):
    - params['fc_0']['kernel'], params['fc_0']['bias']
    - params['fc_1']['kernel'], params['fc_1']['bias']
    - params['fc_2']['kernel'], params['fc_2']['bias']
    - params['fc_3']['kernel'], params['fc_3']['bias']

    Args:
        pytorch_state_dict: PyTorch state_dict from checkpoint
        obs_size: Observation dimension
        action_size: Action dimension
        hidden_dims: Hidden layer dimensions

    Returns:
        actor_params: JAX parameters for actor network
        action_std: Action standard deviation
        obs_normalizer: Observation normalizer
    """
    # Convert actor network weights (only policy, no critic)
    jax_params = {'params': {}}

    # Map PyTorch layer indices to JAX layer names
    # PyTorch uses even indices for linear layers (0, 2, 4, 6, ...)
    # JAX uses fc_0, fc_1, fc_2, fc_3
    pytorch_to_jax_idx = {
        0: 'fc_0',  # First hidden layer
        2: 'fc_1',  # Second hidden layer
        4: 'fc_2',  # Third hidden layer
        6: 'fc_3',  # Output layer
    }

    for pt_idx, jax_name in pytorch_to_jax_idx.items():
        # Get PyTorch weights (they're transposed compared to JAX)
        pt_weight = pytorch_state_dict[f'actor.{pt_idx}.weight'].numpy()  # [out, in]
        pt_bias = pytorch_state_dict[f'actor.{pt_idx}.bias'].numpy()      # [out]

        # JAX Dense uses [in, out] for kernel
        jax_kernel = pt_weight.T  # Transpose to [in, out]
        jax_bias = pt_bias

        jax_params['params'][jax_name] = {
            'kernel': jnp.array(jax_kernel),
            'bias': jnp.array(jax_bias),
        }

    # Extract action std
    if 'std' in pytorch_state_dict:
        action_std = jnp.array(pytorch_state_dict['std'].numpy())
    elif 'log_std' in pytorch_state_dict:
        action_std = jnp.exp(jnp.array(pytorch_state_dict['log_std'].numpy()))
    else:
        print("Warning: No std or log_std found in checkpoint, using default std=1.0")
        action_std = jnp.ones(action_size)

    # Extract observation normalizer (handle both with/without underscore prefix)
    mean_key = None
    std_key = None

    # Check for keys with or without underscore prefix
    if 'actor_obs_normalizer.mean' in pytorch_state_dict:
        mean_key = 'actor_obs_normalizer.mean'
        std_key = 'actor_obs_normalizer.std'
    elif 'actor_obs_normalizer._mean' in pytorch_state_dict:
        mean_key = 'actor_obs_normalizer._mean'
        std_key = 'actor_obs_normalizer._std'

    if mean_key and std_key:
        obs_mean = jnp.array(pytorch_state_dict[mean_key].numpy())
        obs_std = jnp.array(pytorch_state_dict[std_key].numpy())

        # Ensure shapes are (obs_size,) not (1, obs_size)
        if obs_mean.ndim > 1:
            obs_mean = obs_mean.squeeze()
        if obs_std.ndim > 1:
            obs_std = obs_std.squeeze()

        obs_normalizer = ObservationNormalizer(mean=obs_mean, std=obs_std)
        print(f"✓ Observation normalizer loaded (keys: {mean_key}, {std_key})")
    else:
        print("Warning: No observation normalizer found, using identity normalization")
        obs_normalizer = ObservationNormalizer(
            mean=jnp.zeros(obs_size),
            std=jnp.ones(obs_size)
        )

    return jax_params, action_std, obs_normalizer


# ============================================================================
# Environment Interaction
# ============================================================================

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

    def __init__(self, teacher, actor_params, action_std, obs_normalizer):
        self.teacher = teacher
        self.actor_params = actor_params
        self.action_std = action_std
        self.obs_normalizer = obs_normalizer
        self.key = jax.random.PRNGKey(0)

        # JIT compile the action function
        self.get_action_jit = jax.jit(
            lambda params, std, obs, k: teacher.get_action(
                params, std, obs_normalizer, obs, k, deterministic=True
            )
        )

    def __call__(self, obs_dict):
        """Get action from observation (compatible with viewer interface)."""
        # Get observation as numpy array
        obs_array = obs_dict['policy'].cpu().numpy()

        # Convert to JAX array
        obs_jax = jnp.array(obs_array)

        # Get action using JAX network
        self.key, subkey = jax.random.split(self.key)
        action_jax = self.get_action_jit(
            self.actor_params, self.action_std, obs_jax, subkey
        )

        # Convert action back to PyTorch tensor for environment
        action_torch = torch.from_numpy(np.array(action_jax))

        return action_torch


def run_test_episodes(
    teacher: TeacherPolicy,
    actor_params: dict,
    action_std: jnp.ndarray,
    obs_normalizer: ObservationNormalizer,
    env,
    num_episodes: int = 3,
    max_steps: int = 100,
):
    """Run test episodes with the teacher policy.

    Args:
        teacher: TeacherPolicy instance
        actor_params: Actor network parameters
        action_std: Action standard deviation
        obs_normalizer: Observation normalizer
        env: mjlab environment
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    print("\n" + "="*80)
    print("Running Test Episodes")
    print("="*80)

    key = jax.random.PRNGKey(0)

    # JIT compile the action function for speed
    get_action_jit = jax.jit(
        lambda params, std, obs, k: teacher.get_action(
            params, std, obs_normalizer, obs, k, deterministic=True
        )
    )

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0

        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"  Initial observation shape: {obs['policy'].shape}")

        for step in range(max_steps):
            # Get observation as numpy array
            obs_array = obs['policy'].cpu().numpy()

            # Convert to JAX array
            obs_jax = jnp.array(obs_array)

            # Get action using JAX network
            key, subkey = jax.random.split(key)
            action_jax = get_action_jit(actor_params, action_std, obs_jax, subkey)

            # Convert action back to PyTorch tensor for environment
            action_torch = torch.from_numpy(np.array(action_jax))

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_torch)

            episode_reward += reward.cpu().numpy().mean()
            episode_length += 1

            if terminated.any() or truncated.any():
                break

        print(f"  Episode length: {episode_length}")
        print(f"  Total reward: {episode_reward:.3f}")
        print(f"  Average reward per step: {episode_reward/episode_length:.3f}")


# ============================================================================
# Main Test Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test JAX teacher network with mjlab checkpoint and environment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to PyTorch checkpoint (.pt file)'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='franka-lift-cube-v0',
        help='Task ID for mjlab environment'
    )
    parser.add_argument(
        '--num-envs',
        type=int,
        default=1,
        help='Number of parallel environments'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=3,
        help='Number of test episodes to run'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=100,
        help='Maximum steps per episode'
    )
    parser.add_argument(
        '--viewer',
        action='store_true',
        help='Enable viser viewer for visualization'
    )

    args = parser.parse_args()

    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print("="*80)
    print("JAX Teacher Network Test")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Task: {args.task}")
    print(f"Num envs: {args.num_envs}")

    # Load PyTorch checkpoint
    print("\n" + "="*80)
    print("Loading PyTorch Checkpoint")
    print("="*80)
    checkpoint = load_pytorch_checkpoint(str(checkpoint_path))
    print(f"✓ Checkpoint loaded")
    print(f"  Keys in checkpoint: {list(checkpoint.keys())}")

    state_dict = checkpoint['model_state_dict']
    print(f"  Keys in model_state_dict: {list(state_dict.keys())[:10]}...")

    # Load mjlab environment
    print("\n" + "="*80)
    print("Loading mjlab Environment")
    print("="*80)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    print(f"✓ Environment loaded")
    print(f"  Device: {device}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Get dimensions from environment
    obs_sample, _ = env.reset()
    obs_size = obs_sample['policy'].shape[-1]
    action_size = env.action_space.shape[-1]

    print(f"  Observation size: {obs_size}")
    print(f"  Action size: {action_size}")

    # Determine hidden dims from checkpoint
    # Check first hidden layer size
    first_hidden_size = state_dict['actor.0.weight'].shape[0]
    second_hidden_size = state_dict['actor.2.weight'].shape[0]
    third_hidden_size = state_dict['actor.4.weight'].shape[0]
    hidden_dims = (first_hidden_size, second_hidden_size, third_hidden_size)

    print(f"  Hidden dims (from checkpoint): {hidden_dims}")

    # Create JAX teacher policy
    print("\n" + "="*80)
    print("Creating JAX Teacher Policy")
    print("="*80)
    teacher = TeacherPolicy(
        obs_size=obs_size,
        action_size=action_size,
        hidden_dims=hidden_dims,
    )
    print(f"✓ Teacher policy created")

    # Convert PyTorch parameters to JAX
    print("\n" + "="*80)
    print("Converting PyTorch Parameters to JAX")
    print("="*80)
    actor_params, action_std, obs_normalizer = convert_pytorch_to_jax_params(
        state_dict,
        obs_size,
        action_size,
        hidden_dims,
    )
    print(f"✓ Parameters converted")
    print(f"  Actor params keys: {list(actor_params['params'].keys())}")
    print(f"  Action std shape: {action_std.shape}")
    print(f"  Action std values: {action_std}")
    print(f"  Obs normalizer mean shape: {obs_normalizer.mean.shape}")
    print(f"  Obs normalizer std shape: {obs_normalizer.std.shape}")

    # Test forward pass
    print("\n" + "="*80)
    print("Testing Forward Pass")
    print("="*80)
    test_obs = jnp.array(obs_sample['policy'].cpu().numpy())
    test_normalized_obs = obs_normalizer.normalize(test_obs)
    test_action = teacher.actor.apply(actor_params, test_normalized_obs)

    print(f"  Input observation shape: {test_obs.shape}")
    print(f"  Normalized observation shape: {test_normalized_obs.shape}")
    print(f"  Output action shape: {test_action.shape}")
    print(f"  Action value range: [{test_action.min():.3f}, {test_action.max():.3f}]")
    print(f"✓ Forward pass successful")

    # Run test episodes or viewer
    if args.viewer:
        print("\n" + "="*80)
        print("Starting Viser Viewer")
        print("="*80)
        print("Open browser to view the visualization")

        from mjlab.viewer import ViserPlayViewer

        # Create policy wrapper
        policy = JAXPolicyWrapper(teacher, actor_params, action_std, obs_normalizer)

        # Create a new env for viewer (needs num_envs=1)
        env_cfg_viewer = load_env_cfg(args.task, play=True)
        env_cfg_viewer.scene.num_envs = 1
        env_base = ManagerBasedRlEnv(cfg=env_cfg_viewer, device=device)

        # Wrap env to add get_observations method for viewer
        env_viewer = EnvWrapper(env_base)

        # Run viewer
        ViserPlayViewer(env_viewer, policy).run()
        env_viewer.close()
    else:
        run_test_episodes(
            teacher=teacher,
            actor_params=actor_params,
            action_std=action_std,
            obs_normalizer=obs_normalizer,
            env=env,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
        )

    # Clean up
    env.close()

    print("\n" + "="*80)
    print("Test Complete! ✓")
    print("="*80)
    if not args.viewer:
        print("\nNext steps:")
        print("1. Verify the rewards are reasonable (not random)")
        print("2. Use --viewer flag to visualize policy in browser")
        print("3. If rewards look good, we can proceed with data collection")
        print("4. Create separate modules for network, checkpoint loading, and data extraction")


if __name__ == "__main__":
    main()
