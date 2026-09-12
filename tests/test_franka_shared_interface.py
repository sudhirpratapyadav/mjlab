"""Pin the shared semantic layout and action units across tasks and modes."""

import numpy as np
import pytest
import torch

from mjlab.continual_distill.classical.shared_interface import (
  legacy_observation,
  normalized_actions,
)
from mjlab.tasks.manipulation.franka_interface import (
  CENTER,
  FIELDS,
  HALF_RANGE,
  LOWER,
  OBS_DIM,
  SLICES,
  UPPER,
  VERSION,
  NormalizedFrankaPositionAction,
  shared_observation,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

TASKS = [name for name in list_tasks() if name.endswith("-Franka")]


@pytest.mark.parametrize("task", TASKS)
def test_every_registered_mode_has_identical_contract(task):
  reference = load_env_cfg("Mjlab-Reach-Target-Franka").actions
  for mode in ({}, {"test": True}, {"play": True}):
    cfg = load_env_cfg(task, **mode)
    assert cfg.actions == reference
    assert tuple(cfg.observations) == ("policy", "critic")
    for group in cfg.observations.values():
      assert tuple(group.terms) == (VERSION,)
      assert group.terms[VERSION].func is shared_observation
      assert group.terms[VERSION].noise is None
    assert cfg.observations["policy"] == cfg.observations["critic"]
  rl = load_rl_cfg(task)
  assert rl.clip_actions == 1.0
  assert rl.policy.init_noise_std == 0.2


def test_layout_is_fixed_and_contiguous():
  assert OBS_DIM == 60
  assert list(SLICES)[0] == "joint_position"
  assert list(SLICES)[-1] == "control_error"
  assert [i for sl in SLICES.values() for i in range(sl.start, sl.stop)] == list(
    range(60)
  )
  assert sum(FIELDS.values()) == 60


def test_action_clips_before_scaling_and_does_not_integrate():
  # Exercise the real action implementation without building a simulation.
  action = object.__new__(NormalizedFrankaPositionAction)
  action._raw_actions = torch.zeros(3, 8)
  action._scale = torch.tensor(HALF_RANGE[:8])
  action._offset = torch.tensor(CENTER[:8])
  values = torch.tensor([[-2.0] * 8, [0.0] * 8, [2.0] * 8])
  action.process_actions(values)
  expected = torch.tensor([LOWER[:8], CENTER[:8], UPPER[:8]])
  torch.testing.assert_close(action._processed_actions, expected)
  action.process_actions(values)
  torch.testing.assert_close(action._processed_actions, expected)
  # Gripper half-open at 0, no saturation across the usable [-1, 1] range.
  torch.testing.assert_close(
    action._processed_actions[:, 7], torch.tensor([0.0, 0.02, 0.04])
  )


@pytest.mark.parametrize(
  "layout,width", [("standard", 60), ("stack", 51), ("reach", 38)]
)
def test_teacher_boundary_restores_legacy_units_and_roles(layout, width):
  obs = np.arange(120, dtype=float).reshape(2, 60) / 100
  default = np.array([0.0, -1.0, 0.0, -1.57079, 0.0, 2.0, 0.785, 0.04, 0.04])
  old = legacy_observation(obs, default, layout)
  assert old.shape == (2, width)
  np.testing.assert_array_equal(old[:, :18], obs[:, :18])
  np.testing.assert_array_equal(old[:, -8:], obs[:, -8:])
  if layout == "standard":
    np.testing.assert_array_equal(old, obs)
  if layout == "reach":
    np.testing.assert_allclose(old[:, 27:30], obs[:, 40:43] + obs[:, 43:46])
  if layout == "stack":
    np.testing.assert_array_equal(old[:, 37:43], obs[:, 40:46])
  for old_grip, new_grip in [(-1, -1), (-0.5, 0), (0, 1), (1, 1)]:
    a = np.zeros((2, 8))
    a[:, 7] = old_grip
    normalized = normalized_actions(a, default)
    np.testing.assert_allclose(normalized[:, 7], new_grip)
    targets = normalized * HALF_RANGE[:8] + CENTER[:8]
    np.testing.assert_allclose(targets[0, :7], default[:7], atol=1e-6)


def test_tool_missing_information_is_not_silently_invented():
  with pytest.raises(ValueError, match="separate tool pose"):
    legacy_observation(np.zeros((2, 60)), np.zeros(9), "tool")
  old = np.arange(138).reshape(2, 69)
  np.testing.assert_array_equal(legacy_observation(old, np.zeros(9), "tool"), old)


def test_same_width_does_not_guess_the_action_contract():
  from mjlab.continual_distill.classical import CLASSICAL_POLICIES
  from mjlab.continual_distill.classical.lift_object import LiftCubeClassicalPolicy

  assert not LiftCubeClassicalPolicy.shared_observations
  assert CLASSICAL_POLICIES["Mjlab-Lift-Cube-Franka"].shared_observations
