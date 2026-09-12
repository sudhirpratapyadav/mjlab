"""Physical reference and release contracts for the teacher refinements."""

import numpy as np

from mjlab.continual_distill.classical.base import FRANKA_ACTION_SCALE
from mjlab.continual_distill.classical.staged_stack import StagedStackPolicy
from mjlab.continual_distill.classical.topple_block import ToppleBlockClassicalPolicy


def test_topple_punch_keeps_world_target_when_hand_moves():
  policy = ToppleBlockClassicalPolicy(1)
  policy._phase[0] = 2
  policy._push_dir_ok[0] = True
  policy._push_dir[0] = [1.0, 0.0, 0.0]
  obs = np.zeros(60)
  obs[34:40] = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
  start = policy._fk(policy.default_qpos)[0]
  waypoint = start + [0.05, 0.0, 0.05]
  policy._anchor[0] = waypoint
  for shoulder_offset in (0.0, 0.03):
    obs[1] = shoulder_offset
    position = policy._fk(policy.default_qpos + obs[:9])[0]
    error, _, _ = policy._target_error(0, obs)
    np.testing.assert_allclose(position + error, waypoint, atol=1e-10)


def test_stack_release_holds_command_despite_observed_sag():
  policy = StagedStackPolicy(1)
  policy._phase[0] = 6
  policy._q_cmd[0] = policy.default_qpos[:7] + 0.02
  command = policy._q_cmd[0].copy()
  for step, sag in ((1, 0.0), (8, -0.05), (15, -0.1)):
    policy._phase_steps[0] = step
    obs = np.zeros(51)
    obs[1] = sag
    action = policy._act_single(0, obs)
    np.testing.assert_allclose(
      action[:7] * FRANKA_ACTION_SCALE + policy.default_qpos[:7],
      command,
      atol=1e-7,
    )
    assert action[7] == 1.0


def test_tool_heading_is_invariant_to_rolling_its_shaft():
  from mjlab.continual_distill.classical.tool_pull import ToolPullClassicalPolicy

  policy = ToolPullClassicalPolicy(1)
  yaw = 0.25
  c, s = np.cos(yaw), np.sin(yaw)
  rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
  for roll in (0.0, np.pi / 2, np.pi, -np.pi / 2):
    c, s = np.cos(roll), np.sin(roll)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    obs = np.zeros(69)
    obs[55:61] = (rz @ rx)[1:].ravel()
    policy.reset()
    policy._observe(0, obs)
    _, shaft, _ = policy._stick_frame(0)
    np.testing.assert_allclose(shaft, [np.cos(yaw), np.sin(yaw), 0.0], atol=1e-8)
