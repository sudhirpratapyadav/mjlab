"""
Utilities for continual distillation: models and checkpoint loading.

This module provides:
- JAX teacher network matching mjlab's actor structure
- PyTorch checkpoint loading and conversion to JAX
- Observation normalization
"""

from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import torch
from flax import linen as nn


# ============================================================================
# JAX Teacher Network (Actor Only)
# ============================================================================

class TeacherActorMLP(nn.Module):
    """
    JAX MLP matching mjlab's actor network structure.

    Architecture:
    - Hidden layers with ELU activation
    - Output layer for action mean
    - Uses Lecun uniform initialization
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
    """Running statistics for observation normalization."""

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

    def get_action_mean(
        self,
        actor_params: dict,
        obs_normalizer: ObservationNormalizer,
        obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Get action mean from observation (for data collection).

        Args:
            actor_params: Actor network parameters
            obs_normalizer: Observation normalizer
            obs: Raw observations

        Returns:
            action_mean: Mean of action distribution
        """
        # Normalize observation
        normalized_obs = obs_normalizer.normalize(obs)

        # Get action mean from actor network
        action_mean = self.actor.apply(actor_params, normalized_obs)

        return action_mean


# ============================================================================
# Student Network (Multi-Head for Continual Learning)
# ============================================================================

class StudentActorMLP(nn.Module):
    """
    Multi-head student MLP for continual learning.

    Architecture matches teacher (ELU activation, Lecun init) but with
    multiple output heads - one per task.
    """
    action_size: int
    num_tasks: int
    hidden_dims: Tuple[int, ...] = (512, 256, 128)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through student MLP.

        Args:
            obs: Normalized observations [batch_size, obs_dim]

        Returns:
            logits: All task heads concatenated [batch_size, 2*action_size*num_tasks]
                   Format: [task0_mean, task0_scale, task1_mean, task1_scale, ...]
        """
        x = obs

        # Shared hidden layers with ELU activation
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(
                hidden_dim,
                name=f'hidden_{i}',
                kernel_init=nn.initializers.lecun_uniform(),
            )(x)
            x = nn.elu(x)

        # Multi-head output layer
        # Each head outputs 2*action_size (mean + scale_params)
        logits = nn.Dense(
            2 * self.action_size * self.num_tasks,
            name=f'hidden_{len(self.hidden_dims)}',
            kernel_init=nn.initializers.lecun_uniform(),
        )(x)

        return logits


class StudentPolicy:
    """
    Multi-head student policy for continual distillation with SI tracking.

    The student shares an MLP trunk across tasks with separate heads per task.
    Uses ELU activation and same architecture as teacher network.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        num_tasks: int,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        min_std: float = 1e-3,
    ):
        self.obs_size = obs_size
        self.action_size = action_size
        self.num_tasks = num_tasks
        self.hidden_dims = hidden_dims
        self.min_std = min_std

        self.network = StudentActorMLP(
            action_size=action_size,
            num_tasks=num_tasks,
            hidden_dims=hidden_dims,
        )

    def init(self, key: jax.random.PRNGKey) -> dict:
        """Initialize network parameters."""
        dummy_obs = jnp.zeros((1, self.obs_size))
        params = self.network.init(key, dummy_obs)
        return params

    def apply(
        self,
        params: dict,
        normalizer_params,
        obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply network to observations (forward pass).

        Args:
            params: Network parameters
            normalizer_params: Dict with 'mean' and 'std' or ObservationNormalizer
            obs: Raw observations [batch_size, obs_dim]

        Returns:
            logits: All task heads [batch_size, 2*action_size*num_tasks]
        """
        # Normalize observations (handle both dict and ObservationNormalizer)
        if isinstance(normalizer_params, dict):
            normalized_obs = (obs - normalizer_params['mean']) / (normalizer_params['std'] + 1e-8)
        else:
            normalized_obs = normalizer_params.normalize(obs)

        # Forward pass through network
        logits = self.network.apply(params, normalized_obs)

        return logits

    def head_logits(
        self,
        params: dict,
        normalizer_params,
        obs: jnp.ndarray,
        task_idx: int,
    ) -> jnp.ndarray:
        """Get logits for specific task head.

        Args:
            params: Network parameters
            normalizer_params: Dict with 'mean' and 'std' or ObservationNormalizer
            obs: Raw observations
            task_idx: Task index (0 to num_tasks-1)

        Returns:
            head_logits: Logits for specified task [batch_size, 2*action_size]
        """
        logits = self.apply(params, normalizer_params, obs)
        head_dim = 2 * self.action_size
        start = task_idx * head_dim
        return jax.lax.dynamic_slice_in_dim(logits, start_index=start, slice_size=head_dim, axis=-1)

    def head_mean_logstd(
        self,
        params: dict,
        normalizer_params,
        obs: jnp.ndarray,
        task_idx: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get mean and logstd for specific task head.

        Args:
            params: Network parameters
            normalizer_params: Dict with 'mean' and 'std' or ObservationNormalizer
            obs: Raw observations
            task_idx: Task index

        Returns:
            Tuple of (mean, logstd)
        """
        head_logits = self.head_logits(params, normalizer_params, obs, task_idx)
        loc, scale_params = jnp.split(head_logits, 2, axis=-1)
        std = jax.nn.softplus(scale_params) + self.min_std
        log_std = jnp.log(std)
        return loc, log_std

    def get_action(
        self,
        params: dict,
        normalizer_params,
        obs: jnp.ndarray,
        key: jax.random.PRNGKey,
        task_idx: int,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Get action for specific task.

        Args:
            params: Network parameters
            normalizer_params: Dict with 'mean' and 'std' or ObservationNormalizer
            obs: Raw observations
            key: JAX random key (only used if not deterministic)
            task_idx: Task index
            deterministic: If True, return mean; else sample

        Returns:
            action: Unbounded action (no tanh applied)
        """
        mean, log_std = self.head_mean_logstd(params, normalizer_params, obs, task_idx)
        if deterministic:
            return mean

        sample = jax.random.normal(key, shape=mean.shape)
        raw_action = mean + jnp.exp(log_std) * sample
        return raw_action

    @staticmethod
    def flatten_tree(tree: dict) -> jnp.ndarray:
        """Flatten an arbitrary parameter/gradient PyTree into a 1-D buffer."""
        from jax.flatten_util import ravel_pytree
        flat, _ = ravel_pytree(tree)
        return flat


# ============================================================================
# PyTorch Checkpoint Loading & Conversion
# ============================================================================

def load_pytorch_checkpoint(checkpoint_path: str) -> dict:
    """Load PyTorch checkpoint from mjlab training.

    Args:
        checkpoint_path: Path to .pt checkpoint file

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
    - actor_obs_normalizer._mean, actor_obs_normalizer._std

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
    else:
        print("Warning: No observation normalizer found, using identity normalization")
        obs_normalizer = ObservationNormalizer(
            mean=jnp.zeros(obs_size),
            std=jnp.ones(obs_size)
        )

    return jax_params, action_std, obs_normalizer


def load_teacher_from_checkpoint(
    checkpoint_path: str,
    obs_size: int,
    action_size: int,
) -> Tuple[TeacherPolicy, dict, jnp.ndarray, ObservationNormalizer]:
    """Load teacher policy from PyTorch checkpoint.

    This is a convenience function that combines checkpoint loading and conversion.

    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pt file)
        obs_size: Observation dimension
        action_size: Action dimension

    Returns:
        teacher: TeacherPolicy instance
        actor_params: JAX actor parameters
        action_std: Action standard deviation
        obs_normalizer: Observation normalizer
    """
    # Validate checkpoint path
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load PyTorch checkpoint
    checkpoint = load_pytorch_checkpoint(str(checkpoint_path))
    state_dict = checkpoint['model_state_dict']

    # Determine hidden dims from checkpoint
    first_hidden_size = state_dict['actor.0.weight'].shape[0]
    second_hidden_size = state_dict['actor.2.weight'].shape[0]
    third_hidden_size = state_dict['actor.4.weight'].shape[0]
    hidden_dims = (first_hidden_size, second_hidden_size, third_hidden_size)

    # Create teacher policy
    teacher = TeacherPolicy(
        obs_size=obs_size,
        action_size=action_size,
        hidden_dims=hidden_dims,
    )

    # Convert PyTorch parameters to JAX
    actor_params, action_std, obs_normalizer = convert_pytorch_to_jax_params(
        state_dict,
        obs_size,
        action_size,
        hidden_dims,
    )

    return teacher, actor_params, action_std, obs_normalizer
