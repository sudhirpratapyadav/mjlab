"""Task registry system for managing environment registration and creation."""

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

if TYPE_CHECKING:
  from mjlab.tasks.manipulation.taxonomy import TaskTaxonomy


@dataclass
class _TaskCfg:
  env_cfg: ManagerBasedRlEnvCfg
  play_env_cfg: ManagerBasedRlEnvCfg
  test_env_cfg: ManagerBasedRlEnvCfg
  rl_cfg: RslRlOnPolicyRunnerCfg
  runner_cls: type | None
  taxonomy: "TaskTaxonomy | None" = None


# Private module-level registry: task_id -> task config.
_REGISTRY: dict[str, _TaskCfg] = {}


def register_mjlab_task(
  task_id: str,
  env_cfg: ManagerBasedRlEnvCfg,
  play_env_cfg: ManagerBasedRlEnvCfg,
  test_env_cfg: ManagerBasedRlEnvCfg,
  rl_cfg: RslRlOnPolicyRunnerCfg,
  runner_cls: type | None = None,
  taxonomy: "TaskTaxonomy | None" = None,
) -> None:
  """Register an environment task.

  Args:
    task_id: Unique task identifier (e.g., "Mjlab-Velocity-Rough-Unitree-Go1").
    env_cfg: Environment configuration used for training.
    play_env_cfg: Environment configuration in "play" mode.
    test_env_cfg: Environment configuration for testing (no corruption, train episode length).
    rl_cfg: RL runner configuration.
    runner_cls: Optional custom runner class. If None, uses OnPolicyRunner.
    taxonomy: Optional benchmark taxonomy tags (embodiment / skill / fragility). Set
      for manipulation-benchmark tasks so they can be queried and sequenced; leave
      None for non-benchmark tasks (velocity, tracking).
  """
  if task_id in _REGISTRY:
    raise ValueError(f"Task '{task_id}' is already registered")
  _REGISTRY[task_id] = _TaskCfg(
    env_cfg, play_env_cfg, test_env_cfg, rl_cfg, runner_cls, taxonomy
  )


def list_tasks() -> list[str]:
  """List all registered task IDs."""
  return sorted(_REGISTRY.keys())


def load_env_cfg(task_name: str, play: bool = False, test: bool = False) -> ManagerBasedRlEnvCfg:
  """Load environment configuration for a task.

  Args:
    task_name: Name of the registered task.
    play: If True, load play config (infinite episode, no corruption).
    test: If True, load test config (train episode length, no corruption).

  Returns a deep copy to prevent mutation of the registered config.
  """
  if test:
    return deepcopy(_REGISTRY[task_name].test_env_cfg)
  elif play:
    return deepcopy(_REGISTRY[task_name].play_env_cfg)
  else:
    return deepcopy(_REGISTRY[task_name].env_cfg)


def load_rl_cfg(task_name: str) -> RslRlOnPolicyRunnerCfg:
  """Load RL configuration for a task.

  Returns a deep copy to prevent mutation of the registered config.
  """
  return deepcopy(_REGISTRY[task_name].rl_cfg)


def load_runner_cls(task_name: str) -> type | None:
  """Load the runner class for a task.

  If None, the default OnPolicyRunner will be used.
  """
  return _REGISTRY[task_name].runner_cls


def load_taxonomy(task_name: str) -> "TaskTaxonomy | None":
  """Load the benchmark taxonomy tags for a task, or None if untagged."""
  return _REGISTRY[task_name].taxonomy


def list_taxonomy() -> dict[str, "TaskTaxonomy"]:
  """Return {task_id: taxonomy} for all tasks that carry taxonomy tags."""
  return {
    task_id: cfg.taxonomy
    for task_id, cfg in _REGISTRY.items()
    if cfg.taxonomy is not None
  }
